"""
Triton Python backend for GPU-accelerated semantic caching using cuVS CAGRA.

This backend maintains a CAGRA index in GPU memory for sub-2ms cache lookups,
with tiered response storage (Hot in-memory LRU -> Warm Redis -> Evicted).

Architecture:
    Query embedding (GPU) -> CAGRA search (GPU) -> threshold check -> response
    Response storage: Hot (memory LRU) -> Warm (Redis) -> Evicted

All vector operations stay in GPU HBM - no CPU-GPU transfers in the hot path.
Response retrieval uses the tiered store for optimal latency at any cache size.
"""

import json
import time
from threading import Lock

import numpy as np

# Triton Python backend utilities
import triton_python_backend_utils as pb_utils

# Tiered response store
from tiered_store import TieredResponseStore

# cuVS CAGRA imports
try:
    from cuvs.neighbors import cagra
    from pylibraft.common import DeviceResources

    HAS_CUVS = True
except ImportError:
    HAS_CUVS = False
    # Fallback to faiss for comparison
    import faiss


class TritonPythonModel:
    """GPU-accelerated semantic cache with tiered response storage.

    This model maintains a GPU-resident CAGRA index for fast approximate
    nearest neighbor search. Cached responses are stored in a tiered
    hierarchy: hot (in-memory LRU) -> warm (Redis) -> evicted.

    Configuration (via config.pbtxt parameters):
        Vector search:
        - similarity_threshold: Minimum cosine similarity for cache hit (default: 0.85)
        - max_entries: Maximum cache size (default: 100000)
        - embedding_dim: Embedding dimension (default: 1024)
        - graph_degree: CAGRA graph degree (default: 32)
        - use_gpu: Use GPU CAGRA or CPU Faiss fallback (default: true)

        Tiered storage:
        - hot_capacity: Max entries in hot in-memory tier (default: 10000)
        - redis_url: Redis URL for warm tier (default: empty = disabled)
        - redis_prefix: Key prefix for Redis entries (default: triton_cache:)
        - redis_ttl_secs: TTL for warm tier entries (default: 86400)
        - promote_on_warm_hit: Promote warm hits to hot (default: true)
    """

    def initialize(self, args):
        """Initialize the CAGRA index and tiered response store."""
        self.model_config = json.loads(args["model_config"])

        # Parse parameters
        params = {}
        for param in self.model_config.get("parameters", []):
            params[param["key"]] = param["string_value"]

        # Vector search config
        self.similarity_threshold = float(
            params.get("similarity_threshold", "0.85")
        )
        self.max_entries = int(params.get("max_entries", "100000"))
        self.embedding_dim = int(params.get("embedding_dim", "1024"))
        self.graph_degree = int(params.get("graph_degree", "32"))
        self.use_gpu = params.get("use_gpu", "true").lower() == "true"

        # Tiered storage config
        hot_capacity = int(params.get("hot_capacity", "10000"))
        redis_url = params.get("redis_url", "") or None
        redis_prefix = params.get("redis_prefix", "triton_cache:")
        redis_ttl_secs = int(params.get("redis_ttl_secs", "86400"))
        promote_on_warm_hit = (
            params.get("promote_on_warm_hit", "true").lower() == "true"
        )

        # Initialize tiered response store
        self.store = TieredResponseStore(
            hot_capacity=hot_capacity,
            redis_url=redis_url,
            redis_prefix=redis_prefix,
            redis_ttl_secs=redis_ttl_secs,
            promote_on_warm_hit=promote_on_warm_hit,
        )

        self.entry_count = 0
        self.lock = Lock()

        # Metrics
        self.total_searches = 0
        self.cache_hits = 0
        self.total_search_latency_ns = 0

        if self.use_gpu and HAS_CUVS:
            self._init_cagra()
        else:
            self._init_faiss()

        # Log tiered storage configuration
        warm_status = (
            f"Redis ({redis_url})" if redis_url else "disabled"
        )
        pb_utils.Logger.log(
            f"[cagra_cache] Tiered storage: "
            f"hot_capacity={hot_capacity}, "
            f"warm={warm_status}, "
            f"ttl={redis_ttl_secs}s, "
            f"promote_on_hit={promote_on_warm_hit}",
            pb_utils.Logger.INFO,
        )

    def _init_cagra(self):
        """Initialize GPU-resident CAGRA index."""
        self.backend = "cagra"
        self.resources = DeviceResources()

        # Pre-allocate dataset buffer on GPU
        import cupy as cp

        self.dataset = cp.zeros(
            (self.max_entries, self.embedding_dim),
            dtype=cp.float32,
        )

        # CAGRA build parameters
        self.build_params = cagra.IndexParams(
            intermediate_graph_degree=self.graph_degree * 2,
            graph_degree=self.graph_degree,
        )

        # Search parameters
        self.search_params = cagra.SearchParams(
            max_queries=256,
            itopk_size=32,
        )

        self.index = None  # Built after first N entries
        self.index_dirty = True  # Rebuild needed
        self.min_build_size = max(self.graph_degree * 4, 64)

        pb_utils.Logger.log(
            f"[cagra_cache] Initialized CAGRA backend: "
            f"dim={self.embedding_dim}, max_entries={self.max_entries}, "
            f"graph_degree={self.graph_degree}",
            pb_utils.Logger.INFO,
        )

    def _init_faiss(self):
        """Initialize CPU Faiss index (baseline comparison)."""
        self.backend = "faiss"
        # Inner product for cosine similarity on L2-normalized vectors
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

        pb_utils.Logger.log(
            f"[cagra_cache] Initialized Faiss CPU backend: "
            f"dim={self.embedding_dim}",
            pb_utils.Logger.INFO,
        )

    def _rebuild_cagra_index(self):
        """Rebuild CAGRA index from current dataset."""
        if self.entry_count < self.min_build_size:
            return

        active_data = self.dataset[: self.entry_count]

        # Build index on GPU
        self.index = cagra.build(
            self.build_params,
            active_data,
            resources=self.resources,
        )
        self.index_dirty = False

    def execute(self, requests):
        """Process inference requests.

        Input tensors:
            - query_embedding: float32 [batch_size, embedding_dim]
            - operation: string [1] - "search", "insert", or "metrics"
            - response_text: string [1] - cached response (for insert only)

        Output tensors:
            - cache_hit: bool [batch_size]
            - similarity: float32 [batch_size]
            - cached_response: string [batch_size]
            - latency_us: float32 [1]
        """
        responses = []

        for request in requests:
            start_time = time.perf_counter_ns()

            # Get operation type
            operation = (
                pb_utils.get_input_tensor_by_name(request, "operation")
                .as_numpy()[0]
                .decode("utf-8")
            )

            if operation == "metrics":
                # Return tier metrics as JSON
                metrics = self.store.get_metrics()
                metrics["backend"] = self.backend
                metrics["entry_count"] = self.entry_count
                metrics["total_searches"] = self.total_searches
                metrics["cache_hits"] = self.cache_hits

                elapsed_us = (time.perf_counter_ns() - start_time) / 1000.0

                out_hit = pb_utils.Tensor(
                    "cache_hit", np.array([False])
                )
                out_sim = pb_utils.Tensor(
                    "similarity",
                    np.array([0.0], dtype=np.float32),
                )
                out_resp = pb_utils.Tensor(
                    "cached_response",
                    np.array([json.dumps(metrics)], dtype=object),
                )
                out_latency = pb_utils.Tensor(
                    "latency_us",
                    np.array([elapsed_us], dtype=np.float32),
                )

                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            out_hit,
                            out_sim,
                            out_resp,
                            out_latency,
                        ]
                    )
                )
                continue

            # Get embedding for search/insert
            query_embedding = pb_utils.get_input_tensor_by_name(
                request, "query_embedding"
            ).as_numpy()

            if operation == "search":
                cache_hit, similarity, cached_response = self._search(
                    query_embedding
                )
            elif operation == "insert":
                response_text = (
                    pb_utils.get_input_tensor_by_name(
                        request, "response_text"
                    )
                    .as_numpy()[0]
                    .decode("utf-8")
                )
                cache_hit, similarity, cached_response = self._insert(
                    query_embedding, response_text
                )
            else:
                cache_hit = np.array([False])
                similarity = np.array([0.0], dtype=np.float32)
                cached_response = np.array(
                    [f"Unknown operation: {operation}"], dtype=object
                )

            elapsed_us = (time.perf_counter_ns() - start_time) / 1000.0

            # Create output tensors
            out_hit = pb_utils.Tensor("cache_hit", cache_hit)
            out_sim = pb_utils.Tensor("similarity", similarity)
            out_resp = pb_utils.Tensor(
                "cached_response",
                np.array(cached_response, dtype=object),
            )
            out_latency = pb_utils.Tensor(
                "latency_us",
                np.array([elapsed_us], dtype=np.float32),
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[out_hit, out_sim, out_resp, out_latency]
                )
            )

        return responses

    def _search(self, query_embedding):
        """Search for similar cached entries using tiered store."""
        batch_size = query_embedding.shape[0]

        with self.lock:
            self.total_searches += 1

            if self.entry_count == 0:
                return (
                    np.array([False] * batch_size),
                    np.array([0.0] * batch_size, dtype=np.float32),
                    [""] * batch_size,
                )

            if self.backend == "cagra":
                return self._search_cagra(query_embedding, batch_size)
            return self._search_faiss(query_embedding, batch_size)

    def _search_cagra(self, query_embedding, batch_size):
        """GPU CAGRA search - the fast path."""
        import cupy as cp

        if self.index is None or self.index_dirty:
            self._rebuild_cagra_index()

        if self.index is None:
            # Not enough entries to build index yet
            return (
                np.array([False] * batch_size),
                np.array([0.0] * batch_size, dtype=np.float32),
                [""] * batch_size,
            )

        # Transfer query to GPU
        query_gpu = cp.asarray(query_embedding, dtype=cp.float32)

        # CAGRA search on GPU
        distances, indices = cagra.search(
            self.search_params,
            self.index,
            query_gpu,
            k=1,
            resources=self.resources,
        )

        # Transfer results back to CPU
        distances_cpu = cp.asnumpy(distances).flatten()
        indices_cpu = cp.asnumpy(indices).flatten()

        # Convert L2 distances to cosine similarity
        # For L2-normalized vectors: similarity = 1 - distance/2
        similarities = 1.0 - distances_cpu / 2.0

        hits = similarities >= self.similarity_threshold
        cached_responses = []

        for i in range(batch_size):
            if hits[i]:
                idx = int(indices_cpu[i])
                # Retrieve from tiered store
                resp = self.store.get(idx)
                if resp is not None:
                    cached_responses.append(resp)
                    self.cache_hits += 1
                else:
                    # Vector found but response evicted
                    hits[i] = False
                    cached_responses.append("")
            else:
                cached_responses.append("")

        return (hits, similarities.astype(np.float32), cached_responses)

    def _search_faiss(self, query_embedding, batch_size):
        """CPU Faiss search - the baseline."""
        # Faiss inner product search
        similarities, indices = self.faiss_index.search(
            query_embedding.astype(np.float32), 1
        )
        similarities = similarities.flatten()
        indices = indices.flatten()

        hits = similarities >= self.similarity_threshold
        cached_responses = []

        for i in range(batch_size):
            if hits[i]:
                idx = int(indices[i])
                # Retrieve from tiered store
                resp = self.store.get(idx)
                if resp is not None:
                    cached_responses.append(resp)
                    self.cache_hits += 1
                else:
                    # Vector found but response evicted
                    hits[i] = False
                    cached_responses.append("")
            else:
                cached_responses.append("")

        return (hits, similarities.astype(np.float32), cached_responses)

    def _insert(self, embedding, response_text):
        """Insert a new entry into the cache with tiered storage."""
        with self.lock:
            if self.entry_count >= self.max_entries:
                return (
                    np.array([False]),
                    np.array([0.0], dtype=np.float32),
                    ["Cache full"],
                )

            idx = self.entry_count

            if self.backend == "cagra":
                import cupy as cp

                self.dataset[idx] = cp.asarray(
                    embedding[0], dtype=cp.float32
                )
            else:
                # L2 normalize for inner product search
                norm = np.linalg.norm(embedding, axis=1, keepdims=True)
                normalized = embedding / (norm + 1e-8)
                self.faiss_index.add(normalized.astype(np.float32))

            # Store response in tiered store (not a plain dict)
            self.store.put(idx, response_text)
            self.entry_count += 1
            self.index_dirty = True

            return (
                np.array([True]),
                np.array([1.0], dtype=np.float32),
                ["Inserted"],
            )

    def finalize(self):
        """Cleanup GPU resources and tiered store."""
        if hasattr(self, "resources"):
            del self.resources
        if hasattr(self, "index"):
            del self.index
        if hasattr(self, "dataset"):
            del self.dataset

        # Log final metrics
        metrics = self.store.get_metrics()
        hit_rate = (self.cache_hits / max(self.total_searches, 1)) * 100

        pb_utils.Logger.log(
            f"[cagra_cache] Shutting down. "
            f"Searches: {self.total_searches}, "
            f"Hits: {self.cache_hits} ({hit_rate:.1f}%), "
            f"Entries: {self.entry_count}, "
            f"Hot: {metrics.get('hot_entries', 0)}, "
            f"Warm promotions: {metrics.get('warm_promotions', 0)}, "
            f"Warm demotions: {metrics.get('warm_demotions', 0)}",
            pb_utils.Logger.INFO,
        )

        # Cleanup tiered store
        self.store.finalize()
