"""
highrize.cache — Compression result cache.

Hashes input content and caches CompressionResult objects so identical
prompts/images are never compressed twice. Saves CPU time in loops/retries.

Backends:
  - "memory"  : dict in-process (default, no deps)
  - "disk"    : shelve file (persists across runs, no deps)
  - "redis"   : redis-py (distributed, multi-process)
"""

import hashlib
import json
import os
import pickle
from typing import Any, Optional
from .models import CompressionResult, Modality


def _hash_content(content: Any) -> str:
    """Stable hash for any content type."""
    if isinstance(content, str):
        raw = content.encode("utf-8")
    elif isinstance(content, bytes):
        raw = content
    else:
        try:
            raw = json.dumps(content, sort_keys=True, default=str).encode()
        except Exception:
            raw = str(content).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


class CompressionCache:
    """
    Cache wrapper around any HighRize compressor call.

    Usage:
        from highrize.cache import CompressionCache
        from highrize import HighRize

        cache = CompressionCache(backend="disk", path="./highrize_cache")
        tp = HighRize(model="gpt-4o")

        # Wrap any compress call:
        result = cache.get_or_compress(tp, "my prompt text")
        # Second call with same text → instant cache hit, zero re-compression
        result = cache.get_or_compress(tp, "my prompt text")

        print(cache.stats())
    """

    def __init__(
        self,
        backend: str = "memory",
        path: str = "./highrize_cache",
        max_memory_entries: int = 10_000,
        redis_url: str = "redis://localhost:6379/0",
    ):
        self.backend = backend
        self._hits = 0
        self._misses = 0

        if backend == "memory":
            self._store: dict = {}
            self._max = max_memory_entries

        elif backend == "disk":
            import shelve
            dir_path = os.path.dirname(os.path.abspath(path))
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            self._shelf = shelve.open(path)

        elif backend == "redis":
            try:
                import redis
                self._redis = redis.from_url(redis_url)
            except ImportError:
                raise ImportError("Install redis-py: pip install redis")

        else:
            raise ValueError(f"Unknown cache backend: {backend!r}. Use 'memory', 'disk', or 'redis'.")

    def get_or_compress(
        self,
        tp,  # HighRize instance
        content: Any,
        modality: Optional[Modality] = None,
        **kwargs,
    ) -> CompressionResult:
        """
        Return cached result if available, otherwise compress and cache.
        All kwargs are forwarded to tp.compress().
        """
        key = _hash_content(content)
        cached = self._get(key)
        if cached is not None:
            self._hits += 1
            return cached

        self._misses += 1
        result = tp.compress(content, modality=modality, **kwargs)
        self._set(key, result)
        return result

    def invalidate(self, content: Any):
        """Remove a specific entry from cache."""
        key = _hash_content(content)
        self._delete(key)

    def clear(self):
        """Wipe the entire cache."""
        if self.backend == "memory":
            self._store.clear()
        elif self.backend == "disk":
            self._shelf.clear()
            self._shelf.sync()
        elif self.backend == "redis":
            self._redis.flushdb()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = round(self._hits / total * 100, 1) if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": hit_rate,
            "backend": self.backend,
        }

    # --- Internal storage ops ---

    def _get(self, key: str) -> Optional[CompressionResult]:
        try:
            if self.backend == "memory":
                return self._store.get(key)
            elif self.backend == "disk":
                return self._shelf.get(key)
            elif self.backend == "redis":
                raw = self._redis.get(f"highrize:{key}")
                return pickle.loads(raw) if raw else None
        except Exception:
            return None

    def _set(self, key: str, result: CompressionResult):
        try:
            if self.backend == "memory":
                if len(self._store) >= self._max:
                    # Simple LRU-ish: drop oldest 10%
                    drop = list(self._store.keys())[: self._max // 10]
                    for k in drop:
                        del self._store[k]
                self._store[key] = result
            elif self.backend == "disk":
                self._shelf[key] = result
                self._shelf.sync()
            elif self.backend == "redis":
                self._redis.setex(
                    f"highrize:{key}",
                    86400,  # 24h TTL
                    pickle.dumps(result),
                )
        except Exception:
            pass  # Cache write failure is always non-fatal

    def _delete(self, key: str):
        try:
            if self.backend == "memory":
                self._store.pop(key, None)
            elif self.backend == "disk":
                if key in self._shelf:
                    del self._shelf[key]
                    self._shelf.sync()
            elif self.backend == "redis":
                self._redis.delete(f"highrize:{key}")
        except Exception:
            pass

    def __repr__(self):
        return f"CompressionCache(backend={self.backend!r}, hits={self._hits}, misses={self._misses})"
