"""
Thread‑Safe Bounded Blocking Queue implementation using only the Python
standard library.  The queue blocks on `put` when full and on `get` when
empty, employing a single `Condition` to coordinate producers and consumers.
"""

from __future__ import annotations

import collections
import threading
from typing import Deque, Generic, Iterable, List, Optional, TypeVar

T = TypeVar('T')


class BoundedBlockingQueue(Generic[T]):
    """
    A fixed‑capacity FIFO queue that blocks on `put` when full and on `get`
    when empty.

    The implementation guarantees:
    * All public methods are thread‑safe.
    * `put` and `get` have O(1) amortised time.
    * The internal storage uses a `collections.deque` with a maximum length.
    """

    __slots__ = ('_maxsize', '_queue', '_cond')

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")
        self._maxsize: int = maxsize
        self._queue: Deque[T] = collections.deque()
        self._cond = threading.Condition()

    def put(self, item: T, timeout: Optional[float] = None) -> None:
        """
        Insert *item* into the queue, blocking until a free slot exists.

        If *timeout* is provided, the method raises `TimeoutError` after the
        timeout expires without acquiring a slot.
        """
        with self._cond:
            if timeout is not None:
                endtime = threading.current_thread().time() + timeout  # type: ignore[attr-defined]
            while len(self._queue) >= self._maxsize:
                if timeout is None:
                    self._cond.wait()
                else:
                    remaining = endtime - threading.current_thread().time()  # type: ignore[attr-defined]
                    if remaining <= 0:
                        raise TimeoutError("put operation timed out")
                    self._cond.wait(remaining)
            self._queue.append(item)
            self._cond.notify_all()

    def get(self, timeout: Optional[float] = None) -> T:
        """
        Remove and return an item from the queue, blocking until an item is
        available.

        If *timeout* is provided, the method raises `TimeoutError` after the
        timeout expires without receiving an item.
        """
        with self._cond:
            if timeout is not None:
                endtime = threading.current_thread().time() + timeout  # type: ignore[attr-defined]
            while not self._queue:
                if timeout is None:
                    self._cond.wait()
                else:
                    remaining = endtime - threading.current_thread().time()  # type: ignore[attr-defined]
                    if remaining <= 0:
                        raise TimeoutError("get operation timed out")
                    self._cond.wait(remaining)
            item = self._queue.popleft()
            self._cond.notify_all()
            return item

    def size(self) -> int:
        """Return the current number of elements in the queue."""
        with self._cond:
            return len(self._queue)

    def __len__(self) -> int:
        return self.size()

    def __iter__(self) -> Iterable[T]:
        """
        Iterate over a snapshot of the queue's contents.  The snapshot is taken
        under the lock to guarantee consistency, but the iterator itself does
        not hold the lock.
        """
        with self._cond:
            snapshot: List[T] = list(self._queue)
        return iter(snapshot)
