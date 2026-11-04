from __future__ import annotations

import threading
from typing import Callable, Generic, Optional, TypeVar, List, Any

T = TypeVar('T')
R = TypeVar('R')


class BoundedBuffer(Generic[T]):
    """
    Thread‑safe bounded buffer (circular queue) using a Condition.
    Supports blocking and timeout‑based put/get operations.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity: int = capacity
        self._buffer: List[Optional[T]] = [None] * capacity
        self._head: int = 0
        self._tail: int = 0
        self._size: int = 0
        self._cond = threading.Condition()

    def put(self, item: T) -> None:
        """Block until a free slot exists, then insert *item*."""
        with self._cond:
            while self._size == self._capacity:
                self._cond.wait()
            self._buffer[self._tail] = item
            self._tail = (self._tail + 1) % self._capacity
            self._size += 1
            self._cond.notify_all()

    def get(self) -> T:
        """Block until an item is available, then remove and return it."""
        with self._cond:
            while self._size == 0:
                self._cond.wait()
            item = self._buffer[self._head]  # type: ignore[assignment]
            self._buffer[self._head] = None
            self._head = (self._head + 1) % self._capacity
            self._size -= 1
            self._cond.notify_all()
            return item

    def try_put(self, item: T, timeout: Optional[float] = None) -> bool:
        """
        Attempt to put *item* into the buffer.
        Returns True on success, False on timeout.
        """
        with self._cond:
            if timeout is None:
                if self._size == self._capacity:
                    return False
                self._buffer[self._tail] = item
                self._tail = (self._tail + 1) % self._capacity
                self._size += 1
                self._cond.notify_all()
                return True
            else:
                end = threading.current_thread().time() + timeout if hasattr(threading.current_thread(), "time") else None
                while self._size == self._capacity:
                    remaining = None if end is None else max(0.0, end - threading.current_thread().time())
                    if remaining == 0.0:
                        return False
                    self._cond.wait(remaining)
                self._buffer[self._tail] = item
                self._tail = (self._tail + 1) % self._capacity
                self._size += 1
                self._cond.notify_all()
                return True

    def try_get(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        Attempt to retrieve an item.
        Returns the item on success, or None on timeout.
        """
        with self._cond:
            if timeout is None:
                if self._size == 0:
                    return None
                item = self._buffer[self._head]  # type: ignore[assignment]
                self._buffer[self._head] = None
                self._head = (self._head + 1) % self._capacity
                self._size -= 1
                self._cond.notify_all()
                return item
            else:
                end = threading.current_thread().time() + timeout if hasattr(threading.current_thread(), "time") else None
                while self._size == 0:
                    remaining = None if end is None else max(0.0, end - threading.current_thread().time())
                    if remaining == 0.0:
                        return None
                    self._cond.wait(remaining)
                item = self._buffer[self._head]  # type: ignore[assignment]
                self._buffer[self._head] = None
                self._head = (self._head + 1) % self._capacity
                self._size -= 1
                self._cond.notify_all()
                return item

    def __len__(self) -> int:
        with self._cond:
            return self._size

    @property
    def capacity(self) -> int:
        return self._capacity


class Future(Generic[R]):
    """
    Minimal Future implementation supporting result retrieval,
    exception propagation, and cancellation checks.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._result: Optional[R] = None
        self._exception: Optional[BaseException] = None
        self._done: bool = False

    def set_result(self, result: R) -> None:
        with self._cond:
            if self._done:
                raise RuntimeError("Future already completed")
            self._result = result
            self._done = True
            self._cond.notify_all()

    def set_exception(self, exc: BaseException) -> None:
        with self._cond:
            if self._done:
                raise RuntimeError("Future already completed")
            self._exception = exc
            self._done = True
            self._cond.notify_all()

    def result(self, timeout: Optional[float] = None) -> R:
        with self._cond:
            if not self._done:
                self._cond.wait(timeout)
            if not self._done:
                raise TimeoutError("Future result not ready")
            if self._exception:
                raise self._exception
            return self._result  # type: ignore[return-value]

    def done(self) -> bool:
        with self._cond:
            return self._done


class ThreadPool:
    """
    Simple fixed‑size thread pool executing callables asynchronously.
    """

    def __init__(self, max_workers: int) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._max_workers = max_workers
        self._tasks: BoundedBuffer[Callable[[], Any]] = BoundedBuffer(max_workers * 2)
        self._workers: List[threading.Thread] = []
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        for i in range(max_workers):
            t = threading.Thread(target=self._worker_loop, name=f"ThreadPool-Worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def _worker_loop(self) -> None:
        while True:
            with self._shutdown_lock:
                if self._shutdown and len(self._tasks) == 0:
                    break
            try:
                fn = self._tasks.get()
            except Exception:
                continue
            try:
                fn()
            except Exception:
                # Worker isolates task exceptions; they are stored in the Future.
                pass

    def submit(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Future[R]:
        if self._shutdown:
            raise RuntimeError("ThreadPool has been shut down")
        future: Future[R] = Future()

        def _run() -> None:
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except BaseException as exc:
                future.set_exception(exc)

        self._tasks.put(_run)
        return future

    def shutdown(self, wait: bool = True) -> None:
        with self._shutdown_lock:
            self._shutdown = True
        # Wake up workers blocked on get()
        for _ in self._workers:
            self._tasks.put(lambda: None)  # dummy task to unblock
        if wait:
            for t in self._workers:
                t.join()
