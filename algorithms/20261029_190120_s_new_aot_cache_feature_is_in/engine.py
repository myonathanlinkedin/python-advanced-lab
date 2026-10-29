from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Generic, TypeVar, List

from .types import CacheEntry, TKey, TValue


class AOTCache(Generic[TKey, TValue]):
    """
    Ahead-Of-Time (AOT) cache with LRU eviction.

    The cache is backed by a user‑provided loader function that computes a value
    for a given key.  Values can be pre‑populated (AOT) for a collection of keys.
    The cache respects a maximum size; when the size is exceeded the least
    recently used entry is evicted.

    All operations are O(1) amortised thanks to ``OrderedDict``.
    """

    def __init__(self, max_size: int, loader: Callable[[TKey], TValue]) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        self._max_size: int = max_size
        self._loader: Callable[[TKey], TValue] = loader
        # OrderedDict preserves insertion order; we move accessed items to the end.
        self._store: OrderedDict[TKey, CacheEntry[TKey, TValue]] = OrderedDict()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def prepopulate(self, keys: List[TKey]) -> None:
        """
        Compute and store values for the supplied keys using the loader.
        Existing entries are left untouched; new entries are added respecting
        the LRU policy.
        """
        for key in keys:
            if key not in self._store:
                value = self._loader(key)
                self._store[key] = CacheEntry(key, value)
                self._store.move_to_end(key)  # mark as most recently used
                self._evict_if_needed()

    def get(self, key: TKey) -> TValue:
        """
        Retrieve the value for *key*.  If the key is not cached, compute it via
        the loader, store it, and return the value.
        """
        if key in self._store:
            entry = self._store.pop(key)  # remove to re‑insert at end
            self._store[key] = entry
            return entry.value
        # Miss – compute, store, possibly evict
        value = self._loader(key)
        self._store[key] = CacheEntry(key, value)
        self._store.move_to_end(key)
        self._evict_if_needed()
        return value

    def set(self, key: TKey, value: TValue) -> None:
        """
        Explicitly set *value* for *key*, overwriting any existing entry.
        """
        self._store[key] = CacheEntry(key, value)
        self._store.move_to_end(key)
        self._evict_if_needed()

    def invalidate(self, key: TKey) -> None:
        """
        Remove *key* from the cache if present.
        """
        self._store.pop(key, None)

    def __len__(self) -> int:
        """Current number of cached entries."""
        return len(self._store)

    def __contains__(self, key: TKey) -> bool:
        """Membership test."""
        return key in self._store

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _evict_if_needed(self) -> None:
        """
        Evict least‑recently used entries until the cache size is within limits.
        """
        while len(self._store) > self._max_size:
            evicted_key, _ = self._store.popitem(last=False)  # pop LRU
            # No side‑effects required for eviction in this pure implementation.
