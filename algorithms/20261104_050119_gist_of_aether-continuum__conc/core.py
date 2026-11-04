"""
core.py
---------

Implementation of a fair readers‑writer lock (ReadWriteLock) using only the
Python standard library. The lock allows multiple concurrent readers or a
single exclusive writer. It provides explicit acquire/release methods as
well as context‑manager helpers for ergonomic usage.

The implementation follows the classic algorithm:

* Readers may proceed when no writer holds the lock and no writers are waiting.
* Writers obtain exclusive access; they are given priority over newly arriving
  readers to avoid writer starvation.

All public methods are type‑annotated and documented. The lock is safe for
use with Python's `threading` module.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from typing import Optional


class ReadWriteLock:
    """
    A fair readers‑writer lock.

    - Multiple readers can hold the lock simultaneously.
    - Only one writer may hold the lock, and while a writer holds it,
      no readers are allowed.
    - Writers are given priority over new readers to prevent writer starvation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._readers: int = 0               # Number of active readers
        self._writer: bool = False           # True if a writer holds the lock
        self._waiting_writers: int = 0       # Writers queued for the lock
        self._cond = threading.Condition(self._lock)

    # --------------------------------------------------------------------- #
    # Public acquire / release API
    # --------------------------------------------------------------------- #
    def acquire_read(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock in read (shared) mode.

        Returns True if the lock was acquired, False on timeout.
        """
        with self._cond:
            end = None if timeout is None else threading.current_thread().time() + timeout
            while self._writer or self._waiting_writers > 0:
                if timeout is not None:
                    remaining = end - threading.current_thread().time()
                    if remaining <= 0:
                        return False
                    self._cond.wait(remaining)
                else:
                    self._cond.wait()
            self._readers += 1
            return True

    def release_read(self) -> None:
        """Release a previously acquired read lock."""
        with self._cond:
            if self._readers == 0:
                raise RuntimeError("release_read() called without a matching acquire_read()")
            self._readers -= 1
            if self._readers == 0:
                # Wake waiting writers (or readers if no writers)
                self._cond.notify_all()

    def acquire_write(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock in write (exclusive) mode.

        Returns True if the lock was acquired, False on timeout.
        """
        with self._cond:
            self._waiting_writers += 1
            try:
                end = None if timeout is None else threading.current_thread().time() + timeout
                while self._writer or self._readers > 0:
                    if timeout is not None:
                        remaining = end - threading.current_thread().time()
                        if remaining <= 0:
                            return False
                        self._cond.wait(remaining)
                    else:
                        self._cond.wait()
                self._writer = True
                return True
            finally:
                self._waiting_writers -= 1

    def release_write(self) -> None:
        """Release a previously acquired write lock."""
        with self._cond:
            if not self._writer:
                raise RuntimeError("release_write() called without a matching acquire_write()")
            self._writer = False
            # Wake up all waiting threads (writers have priority in acquire_* methods)
            self._cond.notify_all()

    # --------------------------------------------------------------------- #
    # Context‑manager helpers
    # --------------------------------------------------------------------- #
    class _ReadContext(AbstractContextManager[None]):
        def __init__(self, lock: "ReadWriteLock") -> None:
            self._lock = lock

        def __enter__(self) -> None:
            acquired = self._lock.acquire_read()
            if not acquired:
                raise RuntimeError("Failed to acquire read lock")
            return None

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            self._lock.release_read()
            return False  # propagate exceptions

    class _WriteContext(AbstractContextManager[None]):
        def __init__(self, lock: "ReadWriteLock") -> None:
            self._lock = lock

        def __enter__(self) -> None:
            acquired = self._lock.acquire_write()
            if not acquired:
                raise RuntimeError("Failed to acquire write lock")
            return None

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            self._lock.release_write()
            return False

    def read_lock(self) -> "_ReadContext":
        """Return a context manager for a read lock."""
        return self._ReadContext(self)

    def write_lock(self) -> "_WriteContext":
        """Return a context manager for a write lock."""
        return self._WriteContext(self)
