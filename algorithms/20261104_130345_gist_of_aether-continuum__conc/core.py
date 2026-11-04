"""
core.py
-------
A deterministic, lock-free concurrency primitive: a bounded, thread-safe
ring buffer (SPSC / MPSC queue) built on the standard library only.

Design invariants
-----------------
* Bounded capacity C >= 1.
* Producer(s) and consumer(s) coordinate via atomic index increments
  (simulated here with a single lock for correctness; the *algorithmic*
  structure mirrors a lock-free ring buffer).
* Time complexity: O(1) amortised per enqueue / dequeue.
* Space complexity: O(C).
* No data loss, no duplication, FIFO ordering preserved.

The module exposes:
  * RingBuffer  – the core data structure.
  * Producer    – a helper that enqueues items.
  * Consumer    – a helper that dequeues items.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """A bounded, thread-safe FIFO ring buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of elements the buffer can hold (>= 1).

    Invariants
    ----------
    * ``0 <= size <= capacity`` at all times.
    * ``enqueue`` blocks (or raises) when full.
    * ``dequeue`` blocks (or raises) when empty.
    * FIFO order is strictly preserved.
    """

    __slots__ = ("_buf", "_cap", "_head", "_tail", "_size", "_lock", "_cv")

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._buf: list[T | None] = [None] * capacity
        self._cap: int = capacity
        self._head: int = 0  # next write index
        self._tail: int = 0  # next read index
        self._size: int = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, item: T, timeout: float | None = None) -> bool:
        """Insert *item* into the buffer.

        Blocks until space is available or *timeout* expires.
        Returns ``True`` on success, ``False`` on timeout.
        """
        with self._cv:
            if timeout is None:
                while self._size == self._cap:
                    self._cv.wait()
            else:
                while self._size == self._cap:
                    if not self._cv.wait(timeout):
                        return False
            self._buf[self._head] = item
            self._head = (self._head + 1) % self._cap
            self._size += 1
            self._cv.notify()  # wake a waiting consumer
            return True

    def dequeue(self, timeout: float | None = None) -> T | None:
        """Remove and return the oldest item.

        Blocks until an item is available or *timeout* expires.
        Returns ``None`` on timeout (buffer still empty).
        """
        with self._cv:
            if timeout is None:
                while self._size == 0:
                    self._cv.wait()
            else:
                while self._size == 0:
                    if not self._cv.wait(timeout):
                        return None
            item = self._buf[self._tail]
            self._buf[self._tail] = None  # release reference
            self._tail = (self._tail + 1) % self._cap
            self._size -= 1
            self._cv.notify()  # wake a waiting producer
            return item  # type: ignore[return-value]

    def peek(self) -> T | None:
        """Return the oldest item without removing it (non-blocking)."""
        with self._lock:
            if self._size == 0:
                return None
            return self._buf[self._tail]

    def __len__(self) -> int:
        with self._lock:
            return self._size

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._size == 0

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._size == self._cap

    def drain(self, max_items: int | None = None) -> list[T]:
        """Remove up to *max_items* (or all) items, returning them in FIFO order."""
        items: list[T] = []
        with self._cv:
            n = self._size if max_items is None else min(max_items, self._size)
            for _ in range(n):
                item = self._buf[self._tail]
                self._buf[self._tail] = None
                self._tail = (self._tail + 1) % self._cap
                self._size -= 1
                items.append(item)  # type: ignore[arg-type]
            if n > 0:
                self._cv.notify_all()
        return items


class Producer(Generic[T]):
    """Thin wrapper that enqueues items from an iterable into a RingBuffer."""

    __slots__ = ("_buf", "_items", "_thread", "_stop")

    def __init__(self, buf: RingBuffer[T], items: Iterable[T]) -> None:
        self._buf = buf
        self._items = list(items)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        for item in self._items:
            if self._stop.is_set():
                break
            self._buf.enqueue(item)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class Consumer(Generic[T]):
    """Thin wrapper that dequeues items and applies a callback."""

    __slots__ = ("_buf", "_callback", "_thread", "_stop", "_collected")

    def __init__(
        self,
        buf: RingBuffer[T],
        callback: Callable[[T], None] | None = None,
    ) -> None:
        self._buf = buf
        self._callback = callback
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._collected: list[T] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._buf.dequeue(timeout=0.1)
            if item is None:
                continue
            self._collected.append(item)
            if self._callback is not None:
                self._callback(item)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def collected(self) -> list[T]:
        return list(self._collected)
