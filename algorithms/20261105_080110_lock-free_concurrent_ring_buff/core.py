"""
Lock‑Free Single‑Producer Single‑Consumer (SPSC) Ring Buffer.

The implementation relies on the CPython Global Interpreter Lock (GIL) to
provide atomicity for simple integer reads/writes and list element
assignments.  Under CPython this yields a practical lock‑free structure for
the SPSC use‑case without any explicit synchronization primitives.

The buffer has a fixed capacity *n* and stores up to *n‑1* elements; one slot
is left unused to differentiate full vs empty states.

Public API
----------
- `RingBuffer(capacity: int)`: create a buffer with the given capacity.
- `push(item) -> bool`: enqueue *item*; returns ``False`` if the buffer is full.
- `pop() -> Optional[T]`: dequeue an item; returns ``None`` if the buffer is empty.
- `is_empty() -> bool`
- `is_full() -> bool`
- `__len__() -> int`
"""

from __future__ import annotations

from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """
    A lock‑free ring buffer for a single producer thread and a single consumer
    thread.  All operations are O(1) time and use O(capacity) space.

    The implementation is deliberately simple and avoids any Python-level
    locks.  It is safe only when exactly one thread calls ``push`` and exactly
    one thread calls ``pop``; concurrent calls to the same method are not
    supported.
    """

    __slots__ = ("_buf", "_capacity", "_mask", "_head", "_tail")

    def __init__(self, capacity: int) -> None:
        """
        Initialise the buffer.

        Parameters
        ----------
        capacity: int
            Desired capacity.  The actual usable slots are ``capacity - 1``.
            ``capacity`` must be a power of two to enable fast modulo via a mask.
        """
        if capacity < 2:
            raise ValueError("capacity must be at least 2")
        if (capacity & (capacity - 1)) != 0:
            raise ValueError("capacity must be a power of two")
        self._capacity: int = capacity
        self._mask: int = capacity - 1
        self._buf: List[Optional[T]] = [None] * capacity
        self._head: int = 0  # index of next element to pop
        self._tail: int = 0  # index of next slot to push into

    # --------------------------------------------------------------------- #
    # Helper properties (read‑only, accessed only by a single thread each)
    # --------------------------------------------------------------------- #
    @property
    def _head_index(self) -> int:
        """Current head index (consumer view)."""
        return self._head & self._mask

    @property
    def _tail_index(self) -> int:
        """Current tail index (producer view)."""
        return self._tail & self._mask

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def push(self, item: T) -> bool:
        """
        Enqueue *item* into the buffer.

        Returns
        -------
        bool
            ``True`` if the item was stored, ``False`` if the buffer is full.
        """
        next_tail = (self._tail + 1) & ((1 << 64) - 1)  # wrap on 64‑bit overflow
        if next_tail & self._mask == self._head_index:
            # Buffer full – the next slot would collide with head.
            return False
        self._buf[self._tail_index] = item
        # The write to the slot must happen before the tail index update.
        self._tail = next_tail
        return True

    def pop(self) -> Optional[T]:
        """
        Dequeue an element.

        Returns
        -------
        Optional[T]
            The next item, or ``None`` if the buffer is empty.
        """
        if self._head == self._tail:
            # Empty
            return None
        item = self._buf[self._head_index]
        # Help GC – not strictly required but avoids retaining references.
        self._buf[self._head_index] = None
        self._head = (self._head + 1) & ((1 << 64) - 1)
        return item

    def is_empty(self) -> bool:
        """Return ``True`` if the buffer contains no elements."""
        return self._head == self._tail

    def is_full(self) -> bool:
        """Return ``True`` if the buffer cannot accept more elements."""
        next_tail = (self._tail + 1) & ((1 << 64) - 1)
        return (next_tail & self._mask) == self._head_index

    def __len__(self) -> int:
        """Number of stored elements."""
        # Tail may have wrapped around; compute distance modulo capacity.
        return (self._tail - self._head) & self._mask

    # --------------------------------------------------------------------- #
    # Debug helpers (not part of the public contract)
    # --------------------------------------------------------------------- #
    def _snapshot(self) -> List[Optional[T]]:
        """Return a shallow copy of the internal storage for debugging."""
        return self._buf.copy()
