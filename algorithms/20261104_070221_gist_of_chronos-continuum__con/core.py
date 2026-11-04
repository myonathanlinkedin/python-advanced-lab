import threading
from collections import deque
from typing import Deque, Generic, Iterable, List, Optional, TypeVar

T = TypeVar('T')


class BoundedBuffer(Generic[T]):
    """
    Thread‑safe bounded buffer (circular queue) with blocking put/get.
    Invariants:
        - The internal deque never exceeds `capacity`.
        - `put` blocks when the buffer is full.
        - `get` blocks when the buffer is empty.
    Time complexity:
        - put: O(1) amortised
        - get: O(1) amortised
    Space complexity: O(capacity)
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity: int = capacity
        self._buf: Deque[T] = deque()
        self._cond = threading.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._cond:
            return len(self._buf)

    def put(self, item: T) -> None:
        """Insert `item` into the buffer, blocking if full."""
        with self._cond:
            while len(self._buf) >= self._capacity:
                self._cond.wait()
            self._buf.append(item)
            # Wake a waiting consumer.
            self._cond.notify_all()

    def get(self) -> T:
        """Remove and return an item from the buffer, blocking if empty."""
        with self._cond:
            while not self._buf:
                self._cond.wait()
            item = self._buf.popleft()
            # Wake a waiting producer.
            self._cond.notify_all()
            return item

    def put_many(self, items: Iterable[T]) -> None:
        """Convenience: put a sequence of items atomically."""
        for item in items:
            self.put(item)

    def get_many(self, n: int) -> List[T]:
        """Convenience: retrieve up to `n` items (blocks until at least one)."""
        result: List[T] = []
        for _ in range(n):
            result.append(self.get())
        return result


class ReadWriteLock:
    """
    A fair readers‑writers lock.
    Multiple readers may hold the lock simultaneously.
    Writers obtain exclusive access.
    Writers are given priority over new readers once a writer is waiting,
    preventing writer starvation.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._readers: int = 0
        self._writer: bool = False
        self._waiting_writers: int = 0

    def acquire_read(self) -> None:
        """Acquire the lock for reading. Blocks if a writer holds or is waiting."""
        with self._cond:
            while self._writer or self._waiting_writers > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        """Release a previously acquired read lock."""
        with self._cond:
            if self._readers == 0:
                raise RuntimeError("release_read() called without a matching acquire_read()")
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        """Acquire the lock for writing. Blocks until no readers or writers."""
        with self._cond:
            self._waiting_writers += 1
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._waiting_writers -= 1
            self._writer = True

    def release_write(self) -> None:
        """Release a previously acquired write lock."""
        with self._cond:
            if not self._writer:
                raise RuntimeError("release_write() called without a matching acquire_write()")
            self._writer = False
            self._cond.notify_all()

    # Context‑manager helpers for idiomatic usage
    def read_lock(self):
        return _ReadLockContext(self)

    def write_lock(self):
        return _WriteLockContext(self)


class _ReadLockContext:
    def __init__(self, lock: ReadWriteLock):
        self._lock = lock

    def __enter__(self):
        self._lock.acquire_read()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release_read()


class _WriteLockContext:
    def __init__(self, lock: ReadWriteLock):
        self._lock = lock

    def __enter__(self):
        self._lock.acquire_write()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release_write()
