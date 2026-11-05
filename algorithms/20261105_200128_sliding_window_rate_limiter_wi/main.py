from __future__ import annotations
import time
from custom_types import TokenBucketConfig
from engine import TokenBucket, DistributedTokenBucket


class MockTimeProvider:
    """Mock time provider for deterministic testing."""
    def __init__(self, start: float = 0.0) -> None:
        self._current: float = start

    def now(self) -> float:
        return self._current

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("Cannot advance negative time")
        self._current += seconds


def _test_basic_bucket() -> None:
    tp = MockTimeProvider()
    cfg = TokenBucketConfig(capacity=5, window_size=10.0)
    bucket = TokenBucket(cfg, time_provider=tp)

    # Consume 5 tokens
    for _ in range(5):
        assert bucket.allow(), "Token should be allowed within capacity"

    # 6th token should be rejected
    assert not bucket.allow(), "Token should be rejected when capacity exceeded"

    # Advance time beyond window to purge old tokens
    tp.advance(11.0)
    assert bucket.allow(), "Token should be allowed after window expiration"
    assert bucket.tokens_available() == 4, "One token consumed, 4 remaining"


def _test_zero_capacity() -> None:
    tp = MockTimeProvider()
    cfg = TokenBucketConfig(capacity=0, window_size=5.0)
    bucket = TokenBucket(cfg, time_provider=tp)
    assert not bucket.allow(), "No tokens should be allowed with zero capacity"
    assert bucket.tokens_available() == 0, "Available tokens should be zero"


def _test_invalid_config() -> None:
    try:
        TokenBucketConfig(capacity=-1, window_size=5.0)
        assert False, "Negative capacity should raise ValueError"
    except ValueError:
        pass

    try:
        TokenBucketConfig(capacity=5, window_size=0)
        assert False, "Zero window should raise ValueError"
    except ValueError:
        pass


def _test_distributed_bucket() -> None:
    tp = MockTimeProvider()
    cfg = TokenBucketConfig(capacity=3, window_size=10.0)
    bucket1 = TokenBucket(cfg, time_provider=tp)
    bucket2 = TokenBucket(cfg, time_provider=tp)
    dist = DistributedTokenBucket([bucket1, bucket2])

    # Total capacity is 6
    for _ in range(6):
        assert dist.allow(), "Token should be allowed within distributed capacity"

    # 7th token should be rejected
    assert not dist.allow(), "Token should be rejected when distributed capacity exceeded"

    # Advance time to expire all tokens
    tp.advance(11.0)
    assert dist.allow(), "Token should be allowed after window expiration"
    assert dist.tokens_available() == 5, "One token consumed, 5 remaining"


def _test_round_robin_distribution() -> None:
    tp = MockTimeProvider()
    cfg = TokenBucketConfig(capacity=1, window_size=10.0)
    bucket1 = TokenBucket(cfg, time_provider=tp)
    bucket2 = TokenBucket(cfg, time_provider=tp)
    dist = DistributedTokenBucket([bucket1, bucket2])

    # First request goes to bucket1
    assert dist.allow(), "First token should be allowed"
    assert bucket1.tokens_available() == 0
    assert bucket2.tokens_available() == 1

    # Second request goes to bucket2
    assert dist.allow(), "Second token should be allowed"
    assert bucket1.tokens_available() == 0
    assert bucket2.tokens_available() == 0

    # Third request should be rejected (both buckets full)
    assert not dist.allow(), "Third token should be rejected"

    # Advance time to free tokens
    tp.advance(11.0)
    assert dist.allow(), "Token should be allowed after expiration"
    assert bucket1.tokens_available() == 0
    assert bucket2.tokens_available() == 1


def _test_reset() -> None:
    tp = MockTimeProvider()
    cfg = TokenBucketConfig(capacity=2, window_size=5.0)
    bucket = TokenBucket(cfg, time_provider=tp)
    assert bucket.allow()
    assert bucket.allow()
    assert not bucket.allow()
    bucket.reset()
    assert bucket.tokens_available() == 2
    assert bucket.allow()


def run_all_tests() -> None:
    _test_basic_bucket()
    _test_zero_capacity()
    _test_invalid_config()
    _test_distributed_bucket()
    _test_round_robin_distribution()
    _test_reset()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()
