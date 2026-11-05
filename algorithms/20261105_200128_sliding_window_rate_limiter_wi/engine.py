from __future__ import annotations
import collections
import time
from typing import Deque, List, Optional

from custom_types import TokenBucketConfig, TimeProvider


class TokenBucket:
    """Sliding window token bucket implementation."""

    def __init__(self, config: TokenBucketConfig, time_provider: Optional[TimeProvider] = None) -> None:
        if config.capacity < 0:
            raise ValueError("capacity must be non-negative")
        if config.window_size <= 0:
            raise ValueError("window_size must be positive")
        self._capacity: int = config.capacity
        self._window: float = config.window_size
        self._time_provider: TimeProvider = time_provider or time
        self._timestamps: Deque[float] = collections.deque()

    def _now(self) -> float:
        return self._time_provider.now()

    def _purge_old(self) -> None:
        cutoff = self._now() - self._window
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def allow(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens. Returns True if successful."""
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        self._purge_old()
        if len(self._timestamps) + tokens <= self._capacity:
            now = self._now()
            for _ in range(tokens):
                self._timestamps.append(now)
            return True
        return False

    def tokens_available(self) -> int:
        """Return number of tokens currently available."""
        self._purge_old()
        return self._capacity - len(self._timestamps)

    def reset(self) -> None:
        """Clear all tokens."""
        self._timestamps.clear()


class DistributedTokenBucket:
    """Aggregates multiple TokenBuckets to enforce a global capacity."""

    def __init__(self, buckets: List[TokenBucket]) -> None:
        if not buckets:
            raise ValueError("At least one bucket required")
        self._buckets: List[TokenBucket] = buckets
        self._round_robin_index: int = 0

    def allow(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens across the distributed buckets."""
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        # Try each bucket in round-robin order
        for _ in range(len(self._buckets)):
            bucket = self._buckets[self._round_robin_index]
            self._round_robin_index = (self._round_robin_index + 1) % len(self._buckets)
            if bucket.allow(tokens):
                return True
        return False

    def tokens_available(self) -> int:
        """Sum of available tokens across all buckets."""
        return sum(bucket.tokens_available() for bucket in self._buckets)

    def reset(self) -> None:
        """Reset all underlying buckets."""
        for bucket in self._buckets:
            bucket.reset()
