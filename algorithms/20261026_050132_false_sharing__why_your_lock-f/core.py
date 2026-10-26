"""
core.py
---------

Thread‑safe counter implementations illustrating false sharing.

Two counters are provided:

* MutexCounter – a single shared integer protected by a global lock.
* ShardedCounter – a lock‑free‑style counter that distributes increments
  across several independent cells (shards).  When `padded=True` each cell
  is padded to the size of a typical CPU cache line (64 bytes) to avoid
  false sharing.

Both counters expose the same public API:

    increment() -> None
    value() -> int          # total accumulated count
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List


# ----------------------------------------------------------------------
# Helper: a padded cell to mitigate false sharing.
# ----------------------------------------------------------------------
CACHE_LINE_SIZE = 64  # bytes – typical modern x86 cache line


@dataclass
class _PaddedCell:
    """
    A mutable integer cell with an associated lock.
    The `padding` attribute occupies at least one cache line so that two
    distinct cells are unlikely to reside on the same line.
    """
    value: int = 0
    lock: threading.Lock = threading.Lock()
    # The padding is a bytes object; its length forces the interpreter to
    # allocate additional memory.  The exact layout is CPython‑dependent,
    # but the intent is clear for educational purposes.
    padding: bytes = b'\x00' * CACHE_LINE_SIZE


# ----------------------------------------------------------------------
# MutexCounter – simple global lock implementation.
# ----------------------------------------------------------------------
class MutexCounter:
    """
    Counter protected by a single `threading.Lock`.  All threads contend
    for the same lock, which can become a scalability bottleneck.
    """

    __slots__ = ('_value', '_lock')

    def __init__(self) -> None:
        self._value: int = 0
        self._lock = threading.Lock()

    def increment(self) -> None:
        """Atomically increase the counter by one."""
        with self._lock:
            self._value += 1

    def value(self) -> int:
        """Return the current total (acquired under lock for consistency)."""
        with self._lock:
            return self._value


# ----------------------------------------------------------------------
# ShardedCounter – lock‑free‑style counter with optional padding.
# ----------------------------------------------------------------------
class ShardedCounter:
    """
    Counter that spreads increments over `num_shards` independent cells.
    Each cell has its own lock, dramatically reducing contention.
    When `padded` is True, each cell is an instance of `_PaddedCell`,
    which includes extra memory to avoid false sharing.
    """

    __slots__ = ('_cells', '_num_shards', '_padded')

    def __init__(self, num_shards: int = 8, padded: bool = False) -> None:
        if num_shards <= 0:
            raise ValueError("num_shards must be a positive integer")
        self._num_shards = num_shards
        self._padded = padded
        if padded:
            self._cells: List[_PaddedCell] = [_PaddedCell() for _ in range(num_shards)]
        else:
            # Unpadded cells are simple objects with a lock and a value.
            @dataclass
            class _Cell:
                value: int = 0
                lock: threading.Lock = threading.Lock()
            self._cells = [_Cell() for _ in range(num_shards)]

    def _cell_for_current_thread(self):
        """Select a shard based on the current thread identifier."""
        idx = threading.get_ident() % self._num_shards
        return self._cells[idx]

    def increment(self) -> None:
        """Increment the shard belonging to the calling thread."""
        cell = self._cell_for_current_thread()
        # Each cell has its own lock; contention is limited to threads that
        # map to the same shard.
        with cell.lock:
            cell.value += 1

    def value(self) -> int:
        """Aggregate the values of all shards."""
        total = 0
        # Summation does not need to be atomic; a tiny race is acceptable for
        # a read‑only snapshot in this educational context.
        for cell in self._cells:
            # Acquire each cell's lock briefly to obtain a consistent snapshot.
            with cell.lock:
                total += cell.value
        return total


# ----------------------------------------------------------------------
# Utility: run a counter with many threads.
# ----------------------------------------------------------------------
def run_counter(counter: object, increments_per_thread: int, thread_count: int) -> None:
    """
    Spawn `thread_count` threads; each thread calls `counter.increment()`
    `increments_per_thread` times.  The function returns when all threads
    have completed.
    """
    if increments_per_thread < 0 or thread_count < 0:
        raise ValueError("increments_per_thread and thread_count must be non‑negative")

    def worker() -> None:
        for _ in range(increments_per_thread):
            counter.increment()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


__all__ = [
    "MutexCounter",
    "ShardedCounter",
    "run_counter",
]
