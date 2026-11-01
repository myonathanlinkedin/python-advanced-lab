"""
High‑Throughput Stratum‑Cache Ring Buffer

A fixed‑size circular buffer that maintains a small “stratum cache” of the most
recently retrieved items.  The cache allows a consumer to obtain items without
advancing the ring‑buffer head, which can reduce contention in high‑throughput
scenarios where the same data may be read repeatedly.

All operations are O(1) and protected by a single `threading.Lock`, making the
structure safe for multiple producers and consumers in CPython (where the GIL
already serialises byte‑code execution).  The implementation uses only the
standard library and is fully type‑annotated.
"""

from __future__ import annotations

import threading
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class StratumCacheRingBuffer(Generic[T]):
    """
    A ring buffer with an auxiliary cache of the most recent items.

    Parameters
    ----------
    capacity: int
        Total number of slots in the ring buffer. Must be a positive integer.
    cache_size: int
        Number of recent items kept in the cache. Must satisfy
        ``0 <= cache_size <= capacity``.
    """

    __slots__ = ("_capacity", "_cache_size", "_buffer", "_head", "_tail",
                 "_count", "_cache", "_lock")

    def __init__(self, capacity: int, cache_size: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not (0 <= cache_size <= capacity):
            raise ValueError("cache_size must be in [0, capacity]")

        self._capacity: int = capacity
        self._cache_size: int = cache_size
        self._buffer: List[Optional[T]] = [None] * capacity
        self._head: int = 0  # index of next item to get
        self._tail: int = 0  # index of next slot to put
        self._count: int = 0  # number of valid items in buffer
        self._cache: List[T] = []  # most‑recently retrieved items (LIFO)
        self._lock = threading.Lock()

    # --------------------------------------------------------------------- #
    # Core buffer operations
    # --------------------------------------------------------------------- #

    def put(self, item: T) -> None:
        """
        Insert *item* into the buffer.  If the buffer is full, raise
        ``BufferError``.
        """
        with self._lock:
            if self._count == self._capacity:
                raise BufferError("Ring buffer is full")
            self._buffer[self._tail] = item
            self._tail = (self._tail + 1) % self._capacity
            self._count += 1

            # Update cache: newest put items are also considered “recent”.
            if self._cache_size:
                self._cache.insert(0, item)
                if len(self._cache) > self._cache_size:
                    self._cache.pop()

    def get(self) -> T:
        """
        Remove and return the oldest item from the buffer.  If the cache
        contains items, they are returned first (LIFO order) without
        advancing the ring‑buffer head.
        """
        with self._lock:
            if self._cache:
                return self._cache.pop(0)  # most recent cached item

            if self._count == 0:
                raise BufferError("Ring buffer is empty")

            item = self._buffer[self._head]  # type: ignore[assignment]
            self._buffer[self._head] = None  # help GC
            self._head = (self._head + 1) % self._capacity
            self._count -= 1
            return item  # type: ignore[return-value]

    def peek(self) -> T:
        """
        Return the oldest item without removing it.  The cache is consulted
        first; if empty, the item at the head of the ring buffer is returned.
        """
        with self._lock:
            if self._cache:
                return self._cache[0]

            if self._count == 0:
                raise BufferError("Ring buffer is empty")
            return self._buffer[self._head]  # type: ignore[return-value]

    # --------------------------------------------------------------------- #
    # Introspection helpers
    # --------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Number of items currently stored in the buffer (excluding cache)."""
        with self._lock:
            return self._count

    def is_full(self) -> bool:
        """Return ``True`` if the buffer cannot accept more items."""
        with self._lock:
            return self._count == self._capacity

    def is_empty(self) -> bool:
        """Return ``True`` if the buffer contains no items."""
        with self._lock:
            return self._count == 0 and not self._cache

    def cache_contents(self) -> List[T]:
        """Return a shallow copy of the current cache (most‑recent first)."""
        with self._lock:
            return list(self._cache)

    def buffer_snapshot(self) -> List[Optional[T]]:
        """
        Return a snapshot of the underlying ring buffer in logical order
        (oldest → newest).  The cache is **not** included.
        """
        with self._lock:
            snapshot: List[Optional[T]] = []
            idx = self._head
            for _ in range(self._count):
                snapshot.append(self._buffer[idx])
                idx = (idx + 1) % self._capacity
            return snapshot

    # --------------------------------------------------------------------- #
    # Representation
    # --------------------------------------------------------------------- #

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"{self.__class__.__name__}(capacity={self._capacity}, "
                f"cache_size={self._cache_size}, "
                f"len={self._count}, cache_len={len(self._cache)})"
            )
