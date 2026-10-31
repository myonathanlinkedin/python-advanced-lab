from __future__ import annotations
from typing import Generic, TypeVar, List, Optional
import threading

T = TypeVar('T')

class RingBuffer(Generic[T]):
    """
    A fixed-size ring buffer (circular queue) implementation.
    Thread-safe via a single lock for simplicity and correctness.
    """
    __slots__ = ('_capacity', '_buffer', '_head', '_tail', '_count', '_lock')

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self._capacity: int = capacity
        self._buffer: List[Optional[T]] = [None] * capacity
        self._head: int = 0   # points to next element to pop
        self._tail: int = 0   # points to next position to push
        self._count: int = 0
        self._lock = threading.Lock()

    def capacity(self) -> int:
        return self._capacity

    def size(self) -> int:
        with self._lock:
            return self._count

    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0

    def is_full(self) -> bool:
        with self._lock:
            return self._count == self._capacity

    def push(self, item: T) -> None:
        with self._lock:
            if self._count == self._capacity:
                raise OverflowError("RingBuffer is full")
            self._buffer[self._tail] = item
            self._tail = (self._tail + 1) % self._capacity
            self._count += 1

    def pop(self) -> T:
        with self._lock:
            if self._count == 0:
                raise IndexError("Pop from empty RingBuffer")
            item = self._buffer[self._head]
            self._buffer[self._head] = None  # help GC
            self._head = (self._head + 1) % self._capacity
            self._count -= 1
            return item  # type: ignore

    def peek(self) -> T:
        with self._lock:
            if self._count == 0:
                raise IndexError("Peek from empty RingBuffer")
            return self._buffer[self._head]  # type: ignore

    def __iter__(self):
        with self._lock:
            idx = self._head
            for _ in range(self._count):
                yield self._buffer[idx]  # type: ignore
                idx = (idx + 1) % self._capacity

    def __len__(self) -> int:
        return self.size()

    def __repr__(self) -> str:
        with self._lock:
            items = list(self)
            return f"RingBuffer(capacity={self._capacity}, items={items})"
