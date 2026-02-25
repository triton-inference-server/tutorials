<!--
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
-->

# GPU-Accelerated Semantic Caching with cuVS CAGRA

In [Part 8](../Part_8-semantic_caching/README.md), we introduced semantic
caching as a strategy to reduce LLM inference costs by reusing responses
for semantically similar queries. That implementation used CPU-based Faiss
for the vector similarity search.

This tutorial extends that approach by moving the vector search to GPU
using [cuVS CAGRA](https://docs.rapids.ai/api/cuvs/stable/), NVIDIA's
GPU-accelerated approximate nearest neighbor library. We show that GPU
search provides dramatic speedups at scale, and demonstrate a tiered
response store that balances latency and capacity.

## Why GPU Search?

The Part 8 tutorial uses `faiss.IndexFlatL2` for exact nearest neighbor
search on CPU. This works well for small caches, but search latency
grows linearly with cache size:

| Cache Size | CPU Faiss (mean) | GPU CAGRA single (mean) | GPU CAGRA batch-64 (mean) |
|------------|-----------------|------------------------|--------------------------|
| 1,000 | 108 us | 1,138 us | 21 us |
| 10,000 | 2,099 us | 1,179 us | 27 us |
| 100,000 | 20,878 us | 1,196 us | 26 us |

*Measured on NVIDIA A10G (24GB), 1024-dim embeddings, 2000 queries.
RAPIDS 26.02, cuVS CAGRA, Python 3.12.*

Key observations:

1. **CAGRA has ~1.1ms fixed overhead** per GPU kernel launch. For small
   caches (1K entries), CPU Faiss is actually faster.

2. **CAGRA scales flat.** Latency barely changes from 1K to 100K entries
   (1,138us to 1,196us), while Faiss CPU grows linearly
   (108us to 20,878us).

3. **Batching is the key.** Amortizing GPU kernel launch across 64
   queries gives 21-27us per query regardless of cache size, achieving
   **817x speedup** over CPU Faiss at 100K entries.

4. **Throughput.** At 100K entries, CPU Faiss delivers 48 QPS. CAGRA
   batch-64 delivers 39,152 QPS.

## Architecture

This tutorial introduces two improvements over Part 8:

1. **GPU-resident vector search** using cuVS CAGRA instead of CPU Faiss
2. **Tiered response storage** with an in-memory LRU hot tier and
   optional Redis warm tier

```
                    Triton Inference Server
+------------------------------------------------------+
|                                                      |
|   +-------------+    +--------------------------+    |
|   |  Embedding   |--->|    cagra_cache (GPU)     |    |
|   |   Model      |    |                          |    |
|   |  (GPU)       |    |  +--------------------+  |    |
|   +-------------+    |  |  cuVS CAGRA Index   |  |    |
|                       |  |  (GPU HBM)          |  |    |
|                       |  +--------------------+  |    |
|                       |                          |    |
|                       |  +--------------------+  |    |
|                       |  | Tiered Response     |  |    |
|                       |  |   Store             |  |    |
|                       |  |  +------+ +------+  |  |    |
|                       |  |  | Hot  | | Warm |  |  |    |
|                       |  |  |(LRU) | |(Redis)|  |  |    |
|                       |  |  +------+ +------+  |  |    |
|                       |  +--------------------+  |    |
|                       +--------------------------+    |
|                                                      |
|   Cache Hit?                                         |
|   +-- YES (sim >= 0.85): Return cached response      |
|   +-- NO:  Forward to LLM, cache response            |
|                                                      |
+------------------------------------------------------+
```

### Tiered Response Storage

The response store uses two tiers:

- **Hot tier** (in-memory LRU): Sub-microsecond access. Stores the
  most recently accessed responses. When full, least-recently-used
  entries are demoted to the warm tier.

- **Warm tier** (Redis, optional): 1-5ms access. Stores demoted
  responses with a configurable TTL. Warm hits are promoted back to
  the hot tier.

This creates a natural temperature gradient: frequently accessed
responses stay in the fast hot tier, while less popular entries
gracefully degrade to Redis before expiring.

## Prerequisites

- NVIDIA GPU (Ampere or newer: A10G, A100, H100)
- CUDA 12.x
- Docker with NVIDIA Container Toolkit
- Python 3.10+

## Quick Start

### Running the Benchmark

The simplest way to validate these results is with the standalone
benchmark script, which requires no Triton setup:

```bash
# Clone this repository
git clone https://github.com/triton-inference-server/tutorials.git
cd tutorials/Conceptual_Guide/Part_9-gpu_semantic_caching

# Install dependencies
pip install numpy faiss-cpu

# CPU-only benchmark (works on any machine)
python artifacts/benchmark.py --faiss-only

# Full CPU vs GPU comparison (requires cuVS)
pip install cuvs-cu12 cupy-cuda12x
python artifacts/benchmark.py
```

### Running with Triton

```bash
# Copy model files to model repository
mkdir -p model_repository/cagra_cache/1
cp artifacts/model.py model_repository/cagra_cache/1/model.py
cp artifacts/tiered_store.py model_repository/cagra_cache/1/tiered_store.py
cp artifacts/config.pbtxt model_repository/cagra_cache/config.pbtxt

# Start Triton with GPU support
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.12-py3 \
  tritonserver --model-repository=/models
```

## How It Works

### GPU Search with cuVS CAGRA

CAGRA (CUDA Approximate Graph-based Nearest Neighbor) builds a
GPU-optimized proximity graph over the cached embeddings. Unlike
Faiss `IndexFlatL2` which performs exhaustive scan, CAGRA traverses
the graph to find approximate nearest neighbors in near-constant
time:

```python
from cuvs.neighbors import cagra
import cupy as cp

# Build index on GPU
dataset_gpu = cp.asarray(embeddings, dtype=cp.float32)
index_params = cagra.IndexParams(
    intermediate_graph_degree=64,
    graph_degree=32
)
index = cagra.build(index_params, dataset_gpu)

# Search on GPU
search_params = cagra.SearchParams(itopk_size=32)
query_gpu = cp.asarray(query_embedding, dtype=cp.float32)
distances, indices = cagra.search(search_params, index, query_gpu, k=1)
```

The search runs entirely in GPU HBM. If embeddings are already
generated on GPU (as they are in a Triton ensemble), no PCIe
transfer is needed in the hot path.

### Distance to Similarity Conversion

CAGRA returns L2 distances. For normalized embeddings, we convert
to cosine similarity:

```python
# For unit-normalized vectors:
# L2_distance = 2 * (1 - cosine_similarity)
# cosine_similarity = 1 - L2_distance / 2
similarity = 1.0 - l2_distance / 2.0
```

### Tiered Storage

The `TieredResponseStore` manages cached responses across two tiers:

```python
from tiered_store import TieredResponseStore

store = TieredResponseStore(
    hot_capacity=10000,         # Max entries in memory
    redis_url="redis://localhost:6379",  # Optional warm tier
    redis_ttl_secs=86400,       # 24h TTL for warm entries
    promote_on_warm_hit=True    # Promote warm hits to hot
)

# Store a response
store.put(entry_index, "cached LLM response")

# Retrieve (checks hot first, then warm)
response = store.get(entry_index)  # Returns None on miss
```

## Configuration

### Vector Search Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | `0.85` | Minimum cosine similarity for cache hit |
| `max_entries` | `100000` | Maximum cached entries |
| `embedding_dim` | `1024` | Embedding dimension |
| `graph_degree` | `32` | CAGRA graph degree (higher = more accurate) |
| `use_gpu` | `true` | Use GPU CAGRA or fall back to CPU Faiss |

### Tiered Storage Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hot_capacity` | `10000` | Max entries in memory LRU |
| `redis_url` | (empty) | Redis URL for warm tier (empty = disabled) |
| `redis_ttl_secs` | `86400` | TTL for warm tier entries |
| `promote_on_warm_hit` | `true` | Promote warm hits to hot tier |

## Benchmark Results

Full benchmark results measured on NVIDIA A10G (g5.xlarge, EKS):

### Latency

| Cache Size | Backend | Mean | P50 | P95 | P99 | P99.9 | QPS |
|-----------|---------|------|-----|-----|-----|-------|-----|
| 1K | faiss-cpu | 108us | 108us | 113us | 128us | 152us | 9,221 |
| 1K | cagra-gpu | 1,138us | 1,137us | 1,155us | 1,168us | 1,231us | 879 |
| 1K | cagra-batch64 | 21us | 20us | 21us | 23us | 69us | 48,094 |
| 10K | faiss-cpu | 2,099us | 2,064us | 2,326us | 2,666us | 2,948us | 476 |
| 10K | cagra-gpu | 1,179us | 1,180us | 1,194us | 1,210us | 1,252us | 848 |
| 10K | cagra-batch64 | 27us | 26us | 28us | 29us | 84us | 37,575 |
| 100K | faiss-cpu | 20,878us | 20,655us | 22,435us | 24,532us | 26,586us | 48 |
| 100K | cagra-gpu | 1,196us | 1,197us | 1,211us | 1,245us | 1,371us | 836 |
| 100K | cagra-batch64 | 26us | 25us | 26us | 27us | 75us | 39,152 |

### Speedup (vs CPU Faiss)

| Cache Size | CAGRA Single | CAGRA Batch-64 |
|-----------|-------------|----------------|
| 1,000 | 0.1x | **5.2x** |
| 10,000 | **1.8x** | **79x** |
| 100,000 | **17.5x** | **817x** |

### When to Use GPU vs CPU

- **< 5K entries**: CPU Faiss is sufficient. GPU kernel launch overhead
  (~1.1ms) dominates at small cache sizes.
- **5K-50K entries**: GPU CAGRA single-query matches or beats CPU.
  Batch mode provides 10-80x speedup.
- **50K+ entries**: GPU CAGRA is strictly superior. CPU Faiss latency
  becomes impractical (>10ms per query).
- **Batch workloads**: Always prefer GPU with batch sizes of 32-128
  for maximum throughput.

## Comparison with Part 8

| Aspect | Part 8 (CPU) | Part 9 (GPU) |
|--------|-------------|-------------|
| Vector search | Faiss IndexFlatL2 | cuVS CAGRA |
| Hardware | CPU | GPU (Ampere+) |
| Search type | Exact | Approximate (>99% recall) |
| Scaling | O(n) linear | ~O(1) constant |
| Response store | theine LRU | Tiered (LRU + Redis) |
| Embedding model | all-MiniLM-L6-v2 (384-dim) | Any (1024-dim default) |
| Deployment | Single process | Docker Compose (+ Redis) |

## Production Considerations

This tutorial demonstrates GPU-accelerated cache search with tiered
response storage. For production deployments, additional considerations
include:

- **Cache warming**: Pre-populate the cache with common queries
- **Multi-tenant isolation**: Separate caches per tenant
- **Monitoring**: Track hit rates, latency percentiles, tier sizes
- **Persistence**: Redis AOF/RDB for warm tier crash recovery
- **Horizontal scaling**: Multiple Triton replicas with shared Redis

For a production implementation with multi-tier caching,
adaptive GPU memory management, multi-tenant isolation, and
enterprise SLA guarantees, see [Synapse](https://worldflowai.com).

## Files

| File | Description |
|------|-------------|
| `artifacts/benchmark.py` | Standalone CPU vs GPU benchmark |
| `artifacts/model.py` | Triton Python backend for cache model |
| `artifacts/tiered_store.py` | Tiered response store (LRU + Redis) |
| `artifacts/config.pbtxt` | Triton model configuration |

## References

- [cuVS CAGRA Documentation](https://docs.rapids.ai/api/cuvs/stable/)
- [Part 8 - Semantic Caching (CPU)](../Part_8-semantic_caching/README.md)
- [Triton Python Backend](https://github.com/triton-inference-server/python_backend)
- [RAPIDS AI](https://rapids.ai/)
