import threading
import collections
import time
import queue
from typing import Deque, Generic, TypeVar, Optional

T = TypeVar('T')


class BoundedBlockingQueue(Generic[T]):
    """
    Thread‑safe bounded blocking queue implemented with a deque and a
    pair of condition variables.  The implementation mirrors the
    semantics of ``queue.Queue`` but is written from first principles
    for educational clarity.
    """

    def __init__(self, maxsize: int):
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        self._maxsize: int = maxsize
        self._queue: Deque[T] = collections.deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def put(self, item: T, block: bool = True, timeout: Optional[float] = None) -> None:
        """
        Put an item into the queue.  If the queue is full, block until a
        free slot becomes available or until *timeout* seconds elapse.
        Raises ``queue.Full`` on timeout.
        """
        with self._not_full:
            if not block:
                if self.full():
                    raise queue.Full
                self._queue.append(item)
                self._not_empty.notify()
                return

            endtime = None if timeout is None else time.monotonic() + timeout
            while self.full():
                remaining = None if endtime is None else endtime - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise queue.Full
                self._not_full.wait(remaining)
            self._queue.append(item)
            self._not_empty.notify()

    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        """
        Remove and return an item from the queue.  If the queue is empty,
        block until an item is available or until *timeout* seconds elapse.
        Raises ``queue.Empty`` on timeout.
        """
        with self._not_empty:
            if not block:
                if self.empty():
                    raise queue.Empty
                item = self._queue.popleft()
                self._not_full.notify()
                return item

            endtime = None if timeout is None else time.monotonic() + timeout
            while self.empty():
                remaining = None if endtime is None else endtime - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise queue.Empty
                self._not_empty.wait(remaining)
            item = self._queue.popleft()
            self._not_full.notify()
            return item

    def qsize(self) -> int:
        """Return the approximate size of the queue (not thread‑safe)."""
        with self._lock:
            return len(self._queue)

    def empty(self) -> bool:
        """Return ``True`` if the queue is empty, ``False`` otherwise."""
        with self._lock:
            return not self._queue

    def full(self) -> bool:
        """Return ``True`` if the queue is full, ``False`` otherwise."""
        with self._lock:
            return len(self._queue) >= self._maxsize

    def maxsize(self) -> int:
        """Return the configured capacity of the queue."""
        return self._maxsize

    # -----------------------------------------------------------------
    # Convenience dunder methods
    # -----------------------------------------------------------------
    def __len__(self) -> int:
        return self.qsize()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(maxsize={self._maxsize}, size={self.qsize()})"
