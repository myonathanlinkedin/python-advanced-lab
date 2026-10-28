from __future__ import annotations

from typing import Generic, TypeVar, Deque, Optional

T = TypeVar('T')

class Full(Exception):
    """Exception raised when a non‑blocking put() is attempted on a full queue,
    or when a timed put() exceeds the given timeout."""
    pass

class Empty(Exception):
    """Exception raised when a non‑blocking get() is attempted on an empty queue,
    or when a timed get() exceeds the given timeout."""
    pass
