from __future__ import annotations

import threading
import collections
import time
from typing import Generic, Deque, Optional

from .types import T, Full, Empty

class BoundedBlockingQueue(Generic[T]):
    """
    A thread‑safe bounded blocking queue implemented with a Condition variable.
    The queue stores at most ``maxsize`` items.  ``put`` blocks when the queue
    is full, ``get`` blocks when the queue is empty.

    The implementation mirrors the behaviour of ``queue.Queue`` from the
    standard library but is written from first principles for pedagogical
    clarity.
    """

    __slots__ = ('_maxsize', '_queue', '_cond')

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")
        self._maxsize: int = maxsize
        self._queue: Deque[T] = collections.deque()
        self._cond = threading.Condition()

    # --------------------------------------------------------------------- #
    # Helper properties (O(1))
    # --------------------------------------------------------------------- #
    def qsize(self) -> int:
        """Return the approximate size of the queue (non‑blocking)."""
        with self._cond:
            return len(self._queue)

    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise."""
        with self._cond:
            return not self._queue

    def full(self) -> bool:
        """Return True if the queue is full, False otherwise."""
        with self._cond:
            return len(self._queue) >= self._maxsize

    # --------------------------------------------------------------------- #
    # Core API
    # --------------------------------------------------------------------- #
    def put(self, item: T, block: bool = True, timeout: Optional[float] = None) -> None:
        """
        Put ``item`` into the queue.

        If ``block`` is True and the queue is full, block until a free slot
        becomes available or until ``timeout`` seconds have elapsed.
        If ``block`` is False, raise ``Full`` immediately when the queue is full.

        ``timeout`` may be ``None`` (wait indefinitely) or a non‑negative
        float.  A timeout of zero is equivalent to ``block=False``.
        """
        if not block:
            timeout = 0.0

        with self._cond:
            if timeout is not None and timeout < 0:
                raise ValueError("timeout must be non‑negative")

            endtime = None if timeout is None else time.monotonic() + timeout
            while len(self._queue) >= self._maxsize:
                if timeout == 0.0:
                    raise Full
                if timeout is None:
                    self._cond.wait()
                else:
                    remaining = endtime - time.monotonic()
                    if remaining <= 0.0:
                        raise Full
                    self._cond.wait(remaining)

            self._queue.append(item)
            # Wake a single waiting consumer.
            self._cond.notify()

    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        """
        Remove and return an item from the queue.

        If ``block`` is True and the queue is empty, block until an item
        becomes available or until ``timeout`` seconds have elapsed.
        If ``block`` is False, raise ``Empty`` immediately when the queue is empty.

        ``timeout`` may be ``None`` (wait indefinitely) or a non‑negative
        float.  A timeout of zero is equivalent to ``block=False``.
        """
        if not block:
            timeout = 0.0

        with self._cond:
            if timeout is not None and timeout < 0:
                raise ValueError("timeout must be non‑negative")

            endtime = None if timeout is None else time.monotonic() + timeout
            while not self._queue:
                if timeout == 0.0:
                    raise Empty
                if timeout is None:
                    self._cond.wait()
                else:
                    remaining = endtime - time.monotonic()
                    if remaining <= 0.0:
                        raise Empty
                    self._cond.wait(remaining)

            item = self._queue.popleft()
            # Wake a single waiting producer.
            self._cond.notify()
            return item

    # --------------------------------------------------------------------- #
    # Pythonic conveniences
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return self.qsize()

    def __repr__(self) -> str:
        with self._cond:
            return f"{self.__class__.__name__}(maxsize={self._maxsize}, size={len(self._queue)})"
