# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
GPU Semantic Cache Benchmark: Faiss CPU vs cuVS CAGRA GPU

Compares vector search performance for semantic cache lookup at
different cache sizes. Measures latency percentiles, throughput,
and hit rates.

Usage:
    # CPU only (no GPU required)
    pip install numpy faiss-cpu
    python benchmark.py --faiss-only

    # Full CPU vs GPU comparison
    pip install numpy faiss-cpu cuvs-cu12 cupy-cuda12x
    python benchmark.py

    # Custom parameters
    python benchmark.py --entries 1000 10000 100000 --queries 2000 --dim 1024
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class LatencyStats:
    mean: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p999: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0

    @classmethod
    def from_array(cls, arr):
        if len(arr) == 0:
            return cls()
        return cls(
            mean=float(np.mean(arr)),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            p999=float(np.percentile(arr, 99.9)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            std=float(np.std(arr)),
        )


@dataclass
class BenchmarkRun:
    backend: str
    cache_size: int
    num_queries: int
    dimension: int
    similarity_threshold: float
    latency: LatencyStats = field(default_factory=LatencyStats)
    throughput_qps: float = 0.0
    hit_rate: float = 0.0
    avg_similarity: float = 0.0
    build_time_ms: float = 0.0


def get_gpu_info():
    """Query GPU name and memory via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            name = parts[0] if parts else "Unknown"
            mem = int(parts[1]) if len(parts) > 1 else 0
            return name, mem
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "Unknown", 0


def generate_dataset(num_entries, dim, seed=42):
    """Generate a normalized random dataset."""
    rng = np.random.RandomState(seed)
    data = rng.randn(num_entries, dim).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-8
    return data / norms


def generate_queries(dataset, num_queries, noise=0.1, seed=123):
    """Generate queries by adding noise to random dataset entries."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=num_queries, replace=True)
    queries = dataset[indices].copy()
    queries += rng.randn(num_queries, dataset.shape[1]).astype(np.float32) * noise
    norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8
    return queries / norms, indices


def bench_faiss_cpu(dataset, queries, threshold, warmup=50):
    """Benchmark Faiss CPU IndexFlatIP (exact inner product search)."""
    import faiss

    dim = dataset.shape[1]
    index = faiss.IndexFlatIP(dim)

    t0 = time.perf_counter()
    index.add(dataset)
    build_ms = (time.perf_counter() - t0) * 1000

    # Warmup
    for i in range(min(warmup, len(queries))):
        index.search(queries[i : i + 1], 1)

    # Benchmark
    latencies = []
    hits = 0
    total_sim = 0.0
    for i in range(len(queries)):
        t = time.perf_counter_ns()
        scores, _ = index.search(queries[i : i + 1], 1)
        latencies.append((time.perf_counter_ns() - t) / 1000.0)
        sim = float(scores[0][0])
        total_sim += sim
        if sim >= threshold:
            hits += 1

    lat_arr = np.array(latencies)
    total_s = lat_arr.sum() / 1e6
    return BenchmarkRun(
        backend="faiss-cpu",
        cache_size=len(dataset),
        num_queries=len(queries),
        dimension=dim,
        similarity_threshold=threshold,
        latency=LatencyStats.from_array(lat_arr),
        throughput_qps=len(queries) / total_s if total_s > 0 else 0,
        hit_rate=hits / len(queries),
        avg_similarity=total_sim / len(queries),
        build_time_ms=build_ms,
    )


def bench_cagra_single(dataset, queries, threshold, graph_degree=32, warmup=50):
    """Benchmark cuVS CAGRA GPU single-query search."""
    from cuvs.neighbors import cagra
    import cupy as cp

    dim = dataset.shape[1]
    dataset_gpu = cp.asarray(dataset, dtype=cp.float32)

    index_params = cagra.IndexParams(
        intermediate_graph_degree=graph_degree * 2,
        graph_degree=graph_degree,
    )
    t0 = time.perf_counter()
    index = cagra.build(index_params, dataset_gpu)
    cp.cuda.Device(0).synchronize()
    build_ms = (time.perf_counter() - t0) * 1000

    search_params = cagra.SearchParams(itopk_size=32)

    # Warmup
    for i in range(min(warmup, len(queries))):
        query_gpu = cp.asarray(queries[i : i + 1], dtype=cp.float32)
        cagra.search(search_params, index, query_gpu, k=1)
        cp.cuda.Device(0).synchronize()

    # Benchmark
    latencies = []
    hits = 0
    total_sim = 0.0
    for i in range(len(queries)):
        query_gpu = cp.asarray(queries[i : i + 1], dtype=cp.float32)
        t = time.perf_counter_ns()
        distances, _ = cagra.search(search_params, index, query_gpu, k=1)
        cp.cuda.Device(0).synchronize()
        latencies.append((time.perf_counter_ns() - t) / 1000.0)
        dist = float(cp.asnumpy(distances)[0][0])
        sim = 1.0 - dist / 2.0
        total_sim += sim
        if sim >= threshold:
            hits += 1

    lat_arr = np.array(latencies)
    total_s = lat_arr.sum() / 1e6
    del index, dataset_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return BenchmarkRun(
        backend="cuvs-cagra-gpu",
        cache_size=len(dataset),
        num_queries=len(queries),
        dimension=dim,
        similarity_threshold=threshold,
        latency=LatencyStats.from_array(lat_arr),
        throughput_qps=len(queries) / total_s if total_s > 0 else 0,
        hit_rate=hits / len(queries),
        avg_similarity=total_sim / len(queries),
        build_time_ms=build_ms,
    )


def bench_cagra_batch(
    dataset, queries, threshold, batch_size=64, graph_degree=32, warmup=5
):
    """Benchmark cuVS CAGRA GPU batch search."""
    from cuvs.neighbors import cagra
    import cupy as cp

    dim = dataset.shape[1]
    dataset_gpu = cp.asarray(dataset, dtype=cp.float32)

    index_params = cagra.IndexParams(
        intermediate_graph_degree=graph_degree * 2,
        graph_degree=graph_degree,
    )
    t0 = time.perf_counter()
    index = cagra.build(index_params, dataset_gpu)
    cp.cuda.Device(0).synchronize()
    build_ms = (time.perf_counter() - t0) * 1000

    search_params = cagra.SearchParams(itopk_size=32)

    # Warmup
    for _ in range(warmup):
        batch_gpu = cp.asarray(queries[:batch_size], dtype=cp.float32)
        cagra.search(search_params, index, batch_gpu, k=1)
        cp.cuda.Device(0).synchronize()

    # Benchmark
    num_queries = len(queries)
    latencies = []
    hits = 0
    total_sim = 0.0
    for start in range(0, num_queries, batch_size):
        end = min(start + batch_size, num_queries)
        batch = queries[start:end]
        batch_gpu = cp.asarray(batch, dtype=cp.float32)

        t = time.perf_counter_ns()
        distances, _ = cagra.search(search_params, index, batch_gpu, k=1)
        cp.cuda.Device(0).synchronize()
        elapsed_us = (time.perf_counter_ns() - t) / 1000.0

        per_query = elapsed_us / len(batch)
        latencies.extend([per_query] * len(batch))

        dists = cp.asnumpy(distances).flatten()
        sims = 1.0 - dists / 2.0
        total_sim += float(np.sum(sims))
        hits += int(np.sum(sims >= threshold))

    lat_arr = np.array(latencies)
    total_s = lat_arr.sum() / 1e6
    del index, dataset_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return BenchmarkRun(
        backend=f"cuvs-cagra-gpu-batch{batch_size}",
        cache_size=len(dataset),
        num_queries=num_queries,
        dimension=dim,
        similarity_threshold=threshold,
        latency=LatencyStats.from_array(lat_arr),
        throughput_qps=num_queries / total_s if total_s > 0 else 0,
        hit_rate=hits / num_queries,
        avg_similarity=total_sim / num_queries,
        build_time_ms=build_ms,
    )


def print_results(all_runs, sizes, dim, num_queries):
    """Print formatted results summary."""
    print("\n" + "=" * 100)
    print("  RESULTS SUMMARY")
    print("=" * 100)
    for size in sizes:
        runs = [r for r in all_runs if r.cache_size == size]
        if not runs:
            continue
        print(f"\n  Cache: {size:,} entries, {dim}-dim, {num_queries} queries")
        print(
            f"  {'Backend':<25} {'Mean':>10} {'P50':>10} {'P95':>10} "
            f"{'P99':>10} {'P99.9':>10} {'QPS':>12} {'Hits':>8}"
        )
        print("  " + "-" * 97)
        faiss_mean = None
        for r in runs:
            if r.backend == "faiss-cpu":
                faiss_mean = r.latency.mean
            speedup = ""
            if faiss_mean and r.backend != "faiss-cpu" and r.latency.mean > 0:
                speedup = f" ({faiss_mean / r.latency.mean:.0f}x)"
            print(
                f"  {r.backend:<25} {r.latency.mean:>8.1f}us "
                f"{r.latency.p50:>8.1f}us {r.latency.p95:>8.1f}us "
                f"{r.latency.p99:>8.1f}us {r.latency.p999:>8.1f}us "
                f"{r.throughput_qps:>10,.0f} {r.hit_rate:>7.1%}{speedup}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Faiss CPU vs cuVS CAGRA GPU for semantic cache search"
    )
    parser.add_argument(
        "--entries",
        nargs="+",
        type=int,
        default=[1000, 10000, 100000],
        help="Cache sizes to benchmark",
    )
    parser.add_argument(
        "--queries", type=int, default=2000, help="Number of queries per benchmark"
    )
    parser.add_argument(
        "--dim", type=int, default=1024, help="Embedding dimension"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85, help="Similarity threshold"
    )
    parser.add_argument(
        "--noise", type=float, default=0.1, help="Query noise level"
    )
    parser.add_argument(
        "--faiss-only", action="store_true", help="Only run Faiss CPU benchmark"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path"
    )
    args = parser.parse_args()

    gpu_name, gpu_mem = get_gpu_info()

    print("=" * 70)
    print("  GPU SEMANTIC CACHE BENCHMARK")
    print("=" * 70)
    print(f"  GPU:       {gpu_name} ({gpu_mem} MB)")
    print(f"  Python:    {platform.python_version()}")
    print(f"  Host:      {platform.node()}")
    print(f"  Dim:       {args.dim}, Queries: {args.queries}")
    print(f"  Threshold: {args.threshold}")
    print("=" * 70)

    # Check for cuVS
    has_cuvs = False
    if not args.faiss_only:
        try:
            from cuvs.neighbors import cagra  # noqa: F401
            import cupy  # noqa: F401

            has_cuvs = True
            print("  cuVS: AVAILABLE")
        except ImportError as e:
            print(f"  cuVS: NOT AVAILABLE ({e})")
            print("  Running Faiss CPU only. Install cuvs-cu12 for GPU benchmarks.")

    try:
        import faiss  # noqa: F401

        print("  Faiss: AVAILABLE")
    except ImportError:
        print("  ERROR: faiss-cpu not installed!")
        sys.exit(1)

    all_runs = []

    for size in args.entries:
        print(f"\n{'=' * 70}")
        print(f"  CACHE SIZE: {size:,}")
        print("=" * 70)

        dataset = generate_dataset(size, args.dim)
        queries, _ = generate_queries(dataset, args.queries, args.noise)

        # Faiss CPU
        print("  [1/3] Faiss CPU...")
        faiss_run = bench_faiss_cpu(dataset, queries, args.threshold)
        all_runs.append(faiss_run)
        print(
            f"    Mean:{faiss_run.latency.mean:.1f}us "
            f"P99:{faiss_run.latency.p99:.1f}us "
            f"QPS:{faiss_run.throughput_qps:,.0f} "
            f"Hits:{faiss_run.hit_rate:.1%}"
        )

        if has_cuvs:
            # CAGRA single-query
            print("  [2/3] CAGRA single-query...")
            cagra_run = bench_cagra_single(dataset, queries, args.threshold)
            all_runs.append(cagra_run)
            speedup = faiss_run.latency.mean / max(cagra_run.latency.mean, 0.001)
            print(
                f"    Mean:{cagra_run.latency.mean:.1f}us "
                f"P99:{cagra_run.latency.p99:.1f}us "
                f"QPS:{cagra_run.throughput_qps:,.0f} "
                f"Speedup:{speedup:.1f}x"
            )

            # CAGRA batch
            print("  [3/3] CAGRA batch-64...")
            batch_run = bench_cagra_batch(dataset, queries, args.threshold)
            all_runs.append(batch_run)
            batch_speedup = faiss_run.latency.mean / max(
                batch_run.latency.mean, 0.001
            )
            print(
                f"    Mean:{batch_run.latency.mean:.1f}us "
                f"P99:{batch_run.latency.p99:.1f}us "
                f"QPS:{batch_run.throughput_qps:,.0f} "
                f"Speedup:{batch_speedup:.1f}x"
            )

    # Print summary
    print_results(all_runs, args.entries, args.dim, args.queries)

    # Build results dict
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "gpu_name": gpu_name,
        "gpu_memory_mb": gpu_mem,
        "python_version": platform.python_version(),
        "config": {
            "dimension": args.dim,
            "num_queries": args.queries,
            "threshold": args.threshold,
            "noise": args.noise,
            "cache_sizes": args.entries,
        },
        "runs": [asdict(r) for r in all_runs],
    }

    # Save results
    output_path = args.output or "/tmp/benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    # Print JSON for machine parsing
    print("\n--- JSON_RESULTS_START ---")
    print(json.dumps(results, indent=2))
    print("--- JSON_RESULTS_END ---")


if __name__ == "__main__":
    main()
