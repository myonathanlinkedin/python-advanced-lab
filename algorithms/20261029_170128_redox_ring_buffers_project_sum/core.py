from __future__ import annotations
from typing import Generic, TypeVar, Iterable, Iterator, List, Optional

T = TypeVar('T')

class RingBuffer(Generic[T]):
    """
    A fixed-size circular buffer (ring buffer) implementation.
    Supports O(1) push and pop operations.
    """
    __slots__ = ('_capacity', '_buffer', '_head', '_tail', '_count')

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self._capacity: int = capacity
        self._buffer: List[Optional[T]] = [None] * capacity
        self._head: int = 0   # points to the oldest element
        self._tail: int = 0   # points to the next free slot
        self._count: int = 0

    def push(self, item: T) -> None:
        """Insert an item into the buffer. Raises OverflowError if full."""
        if self.is_full():
            raise OverflowError("RingBuffer is full")
        self._buffer[self._tail] = item
        self._tail = (self._tail + 1) % self._capacity
        self._count += 1

    def pop(self) -> T:
        """Remove and return the oldest item. Raises IndexError if empty."""
        if self.is_empty():
            raise IndexError("Pop from empty RingBuffer")
        item = self._buffer[self._head]
        self._buffer[self._head] = None  # help GC
        self._head = (self._head + 1) % self._capacity
        self._count -= 1
        return item  # type: ignore

    def peek(self) -> T:
        """Return the oldest item without removing it. Raises IndexError if empty."""
        if self.is_empty():
            raise IndexError("Peek from empty RingBuffer")
        return self._buffer[self._head]  # type: ignore

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return self._count == 0

    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return self._count == self._capacity

    def size(self) -> int:
        """Return the number of items currently stored."""
        return self._count

    def clear(self) -> None:
        """Remove all items from the buffer."""
        self._buffer = [None] * self._capacity
        self._head = 0
        self._tail = 0
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[T]:
        """Iterate over items from oldest to newest."""
        idx = self._head
        for _ in range(self._count):
            yield self._buffer[idx]  # type: ignore
            idx = (idx + 1) % self._capacity

    def to_list(self) -> List[T]:
        """Return a list of items from oldest to newest."""
        return list(self)

    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self._capacity}, items={self.to_list()})"

    def extend(self, items: Iterable[T]) -> None:
        """Add multiple items to the buffer. Raises OverflowError if not enough space."""
        for item in items:
            self.push(item)

    def __getitem__(self, index: int) -> T:
        """
        Get item by index relative to the oldest element.
        Negative indices are supported.
        """
        if not 0 <= index < self._count and not -self._count <= index < 0:
            raise IndexError("Index out of range")
        if index < 0:
            index += self._count
        actual_index = (self._head + index) % self._capacity
        return self._buffer[actual_index]  # type: ignore
