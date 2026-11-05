from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Deque, Iterable, List, Optional


class TimeProvider(Protocol):
    """Protocol for time source abstraction."""
    def now(self) -> float:
        """Return current time in seconds as a float."""


@dataclass(frozen=True)
class TokenBucketConfig:
    """Configuration for a token bucket."""
    capacity: int          # Maximum number of tokens in the bucket
    window_size: float     # Sliding window size in seconds
