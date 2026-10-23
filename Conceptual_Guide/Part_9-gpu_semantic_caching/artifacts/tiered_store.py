"""
Tiered Response Store for GPU-accelerated semantic caching.

Implements a three-tier response storage hierarchy:

    Hot (In-Memory LRU)  ->  Warm (Redis)  ->  Evicted

Hot tier:
    - OrderedDict with LRU eviction
    - Sub-microsecond access for frequently-used responses
    - Configurable max capacity (default: 10,000 entries)

Warm tier:
    - Redis with configurable TTL
    - 1-5ms access latency
    - Horizontal scaling via Redis Cluster
    - Entries promoted to Hot on access

Eviction flow:
    - New inserts go to Hot tier
    - When Hot tier is full, LRU entry demotes to Warm (Redis)
    - When Warm tier TTL expires, entry is evicted entirely
    - On cache hit from Warm tier, entry is promoted back to Hot

Access tracking:
    - Each entry tracks access count and last access time
    - Eviction score = 1 / (access_count * recency_weight)
    - Lower score = more likely to be evicted

Architecture:
    +-----------+     demote     +-----------+     expire    +-----------+
    |   Hot     | ------------> |   Warm    | ------------> |  Evicted  |
    | (Memory)  | <------------ |  (Redis)  |               |  (Gone)   |
    +-----------+    promote    +-----------+               +-----------+
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StoreEntry:
    """A cached response with access metadata."""

    response: str
    access_count: int = 0
    last_access_time: float = 0.0
    created_time: float = 0.0
    size_bytes: int = 0

    def touch(self) -> None:
        """Record an access to this entry."""
        self.access_count += 1
        self.last_access_time = time.monotonic()

    def to_dict(self) -> dict:
        """Serialize for Redis storage."""
        return {
            "response": self.response,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "created_time": self.created_time,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StoreEntry:
        """Deserialize from Redis storage."""
        return cls(
            response=data["response"],
            access_count=data.get("access_count", 0),
            last_access_time=data.get("last_access_time", 0.0),
            created_time=data.get("created_time", 0.0),
            size_bytes=data.get("size_bytes", 0),
        )


@dataclass
class TierMetrics:
    """Per-tier performance metrics."""

    gets: int = 0
    hits: int = 0
    puts: int = 0
    evictions: int = 0
    promotions: int = 0
    demotions: int = 0
    total_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.gets, 1)


@dataclass
class StoreMetrics:
    """Aggregate metrics across all tiers."""

    hot: TierMetrics = field(default_factory=TierMetrics)
    warm: TierMetrics = field(default_factory=TierMetrics)
    total_gets: int = 0
    total_hits: int = 0

    @property
    def overall_hit_rate(self) -> float:
        return self.total_hits / max(self.total_gets, 1)

    def summary(self) -> dict:
        """Return a serializable summary."""
        return {
            "hot_entries": self.hot.puts - self.hot.evictions,
            "hot_hit_rate": f"{self.hot.hit_rate:.2%}",
            "hot_gets": self.hot.gets,
            "hot_hits": self.hot.hits,
            "hot_evictions": self.hot.evictions,
            "warm_entries": self.warm.puts - self.warm.evictions,
            "warm_hit_rate": f"{self.warm.hit_rate:.2%}",
            "warm_gets": self.warm.gets,
            "warm_hits": self.warm.hits,
            "warm_promotions": self.warm.promotions,
            "warm_demotions": self.warm.demotions,
            "overall_hit_rate": f"{self.overall_hit_rate:.2%}",
            "total_gets": self.total_gets,
            "total_hits": self.total_hits,
        }


class TieredResponseStore:
    """Three-tier response store: Hot (memory) -> Warm (Redis) -> Evicted.

    Thread-safe via a single lock. The lock scope is kept minimal:
    hot-tier dict operations are O(1) and Redis calls happen outside
    the lock when possible.

    Parameters:
        hot_capacity: Maximum entries in the hot (in-memory) tier.
        redis_url: Redis connection URL (e.g., "redis://localhost:6379").
            If None, warm tier is disabled and evicted entries are lost.
        redis_prefix: Key prefix for Redis entries (for namespace isolation).
        redis_ttl_secs: TTL for warm-tier entries in Redis.
        promote_on_warm_hit: Whether to promote entries from warm to hot
            on access (default: True).
    """

    def __init__(
        self,
        hot_capacity: int = 10_000,
        redis_url: Optional[str] = None,
        redis_prefix: str = "triton_cache:",
        redis_ttl_secs: int = 86400,
        promote_on_warm_hit: bool = True,
    ):
        self._hot_capacity = hot_capacity
        self._redis_prefix = redis_prefix
        self._redis_ttl_secs = redis_ttl_secs
        self._promote_on_warm_hit = promote_on_warm_hit
        self._lock = Lock()

        # Hot tier: OrderedDict for O(1) LRU
        # move_to_end() on access, popitem(last=False) for eviction
        self._hot: OrderedDict[int, StoreEntry] = OrderedDict()

        # Warm tier: Redis connection (lazy)
        self._redis = None
        self._redis_url = redis_url
        if redis_url:
            self._connect_redis(redis_url)

        # Metrics
        self.metrics = StoreMetrics()

    def _connect_redis(self, url: str) -> None:
        """Establish Redis connection."""
        try:
            import redis

            self._redis = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=2,
                retry_on_timeout=True,
            )
            # Test connection
            self._redis.ping()
            logger.info("Warm tier connected: %s", url)
        except ImportError:
            logger.warning(
                "redis package not installed. Warm tier disabled. "
                "Install with: pip install redis"
            )
            self._redis = None
        except Exception as e:
            logger.warning(
                "Failed to connect to Redis at %s: %s. "
                "Warm tier disabled.",
                url,
                e,
            )
            self._redis = None

    def get(self, idx: int) -> Optional[str]:
        """Retrieve a cached response by index.

        Checks hot tier first, then warm tier. Promotes warm hits
        to hot tier if configured.

        Returns:
            The cached response string, or None if not found.
        """
        self.metrics.total_gets += 1

        # Check hot tier
        with self._lock:
            self.metrics.hot.gets += 1
            entry = self._hot.get(idx)
            if entry is not None:
                entry.touch()
                self._hot.move_to_end(idx)
                self.metrics.hot.hits += 1
                self.metrics.total_hits += 1
                return entry.response

        # Check warm tier (Redis) - outside lock for I/O
        if self._redis is not None:
            self.metrics.warm.gets += 1
            warm_entry = self._get_from_redis(idx)
            if warm_entry is not None:
                self.metrics.warm.hits += 1
                self.metrics.total_hits += 1

                # Promote to hot tier
                if self._promote_on_warm_hit:
                    warm_entry.touch()
                    self._promote(idx, warm_entry)

                return warm_entry.response

        return None

    def put(self, idx: int, response: str) -> None:
        """Store a response in the hot tier.

        If the hot tier is full, the LRU entry is demoted to
        the warm tier (Redis) before insertion.

        Args:
            idx: Cache index (maps to CAGRA index position).
            response: The cached response string.
        """
        now = time.monotonic()
        entry = StoreEntry(
            response=response,
            access_count=1,
            last_access_time=now,
            created_time=now,
            size_bytes=len(response.encode("utf-8")),
        )

        demoted_idx = None
        demoted_entry = None

        with self._lock:
            # If already in hot, just update
            if idx in self._hot:
                self._hot[idx] = entry
                self._hot.move_to_end(idx)
                return

            # Evict LRU if at capacity
            if len(self._hot) >= self._hot_capacity:
                demoted_idx, demoted_entry = self._hot.popitem(last=False)
                self.metrics.hot.evictions += 1

            self._hot[idx] = entry
            self._hot.move_to_end(idx)
            self.metrics.hot.puts += 1

        # Demote evicted entry to warm tier (outside lock)
        if demoted_entry is not None and self._redis is not None:
            self._demote(demoted_idx, demoted_entry)

    def delete(self, idx: int) -> bool:
        """Remove an entry from all tiers.

        Returns:
            True if the entry was found and removed.
        """
        found = False

        with self._lock:
            if idx in self._hot:
                del self._hot[idx]
                found = True

        if self._redis is not None:
            key = f"{self._redis_prefix}{idx}"
            try:
                if self._redis.delete(key):
                    found = True
            except Exception as e:
                logger.warning("Redis delete failed for %s: %s", key, e)

        return found

    def clear(self) -> None:
        """Remove all entries from all tiers."""
        with self._lock:
            self._hot.clear()

        if self._redis is not None:
            self._clear_redis()

    @property
    def hot_size(self) -> int:
        """Number of entries in the hot tier."""
        return len(self._hot)

    @property
    def hot_capacity(self) -> int:
        """Maximum hot tier capacity."""
        return self._hot_capacity

    @property
    def has_warm_tier(self) -> bool:
        """Whether the warm (Redis) tier is available."""
        return self._redis is not None

    def warm_size(self) -> int:
        """Approximate number of entries in the warm tier.

        Uses Redis DBSIZE which includes all keys, so this is only
        accurate when the Redis instance is dedicated to the cache.
        """
        if self._redis is None:
            return 0
        try:
            # Count keys matching our prefix
            count = 0
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(
                    cursor=cursor,
                    match=f"{self._redis_prefix}*",
                    count=1000,
                )
                count += len(keys)
                if cursor == 0:
                    break
            return count
        except Exception:
            return 0

    def get_metrics(self) -> dict:
        """Return current metrics as a serializable dict."""
        return self.metrics.summary()

    # ---------------------------------------------------------------
    # Internal: Redis operations
    # ---------------------------------------------------------------

    def _get_from_redis(self, idx: int) -> Optional[StoreEntry]:
        """Fetch an entry from Redis warm tier."""
        key = f"{self._redis_prefix}{idx}"
        try:
            data = self._redis.get(key)
            if data is None:
                return None
            return StoreEntry.from_dict(json.loads(data))
        except Exception as e:
            logger.warning("Redis get failed for %s: %s", key, e)
            return None

    def _demote(self, idx: int, entry: StoreEntry) -> None:
        """Demote an entry from hot to warm (Redis)."""
        key = f"{self._redis_prefix}{idx}"
        try:
            self._redis.setex(
                key,
                self._redis_ttl_secs,
                json.dumps(entry.to_dict()),
            )
            self.metrics.warm.puts += 1
            self.metrics.warm.demotions += 1
            logger.debug(
                "Demoted entry %d to warm tier (accesses=%d)",
                idx,
                entry.access_count,
            )
        except Exception as e:
            logger.warning("Redis demote failed for %s: %s", key, e)

    def _promote(self, idx: int, entry: StoreEntry) -> None:
        """Promote an entry from warm (Redis) to hot."""
        demoted_idx = None
        demoted_entry = None

        with self._lock:
            # Evict LRU from hot if at capacity
            if len(self._hot) >= self._hot_capacity:
                demoted_idx, demoted_entry = self._hot.popitem(last=False)
                self.metrics.hot.evictions += 1

            self._hot[idx] = entry
            self._hot.move_to_end(idx)
            self.metrics.warm.promotions += 1

        # Remove promoted entry from Redis
        key = f"{self._redis_prefix}{idx}"
        try:
            self._redis.delete(key)
        except Exception as e:
            logger.warning("Redis delete (promote) failed for %s: %s", key, e)

        # Demote the evicted hot entry to warm
        if demoted_entry is not None:
            self._demote(demoted_idx, demoted_entry)

    def _clear_redis(self) -> None:
        """Remove all cache entries from Redis."""
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(
                    cursor=cursor,
                    match=f"{self._redis_prefix}*",
                    count=1000,
                )
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning("Redis clear failed: %s", e)

    def finalize(self) -> None:
        """Clean up resources."""
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass
            self._redis = None
