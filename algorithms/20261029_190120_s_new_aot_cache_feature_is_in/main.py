from __future__ import annotations

import sys
from typing import List

# Import from sibling modules using relative imports
from .engine import AOTCache
from .types import CacheEntry


def _factorial(n: int) -> int:
    """Simple recursive factorial for demonstration."""
    if n < 0:
        raise ValueError("factorial undefined for negative numbers")
    return 1 if n == 0 else n * _factorial(n - 1)


def _loader_factory(counter: dict) -> callable:
    """
    Returns a loader that increments *counter['calls']* each time it is invoked.
    Useful for asserting lazy evaluation.
    """
    def loader(x: int) -> int:
        counter['calls'] += 1
        return _factorial(x)
    return loader


def run_unit_tests() -> None:
    """
    Exhaustive assertions covering typical and edge‑case behaviour.
    """
    # --------------------------------------------------------------------- #
    # Test 1: Basic get/set functionality
    # --------------------------------------------------------------------- #
    call_counter = {'calls': 0}
    cache = AOTCache[int, int](max_size=5, loader=_loader_factory(call_counter))

    # Cache miss – loader should be called
    assert cache.get(4) == 24
    assert call_counter['calls'] == 1
    assert len(cache) == 1
    assert 4 in cache

    # Cache hit – loader must NOT be called again
    prev_calls = call_counter['calls']
    assert cache.get(4) == 24
    assert call_counter['calls'] == prev_calls

    # Explicit set overrides existing value
    cache.set(4, 999)
    assert cache.get(4) == 999
    # Loader not invoked for overridden entry
    assert call_counter['calls'] == prev_calls

    # --------------------------------------------------------------------- #
    # Test 2: Prepopulate and LRU eviction
    # --------------------------------------------------------------------- #
    call_counter['calls'] = 0
    cache = AOTCache[int, int](max_size=3, loader=_loader_factory(call_counter))

    cache.prepopulate([1, 2, 3])
    assert len(cache) == 3
    assert call_counter['calls'] == 3
    assert cache.get(2) == 2  # 2! = 2
    # Access order now: 1 (LRU), 3, 2 (MRU)

    # Adding a new key forces eviction of LRU (key 1)
    cache.get(4)  # computes 24
    assert len(cache) == 3
    assert 1 not in cache
    assert 2 in cache and 3 in cache and 4 in cache

    # --------------------------------------------------------------------- #
    # Test 3: Invalidate removes entry without side‑effects
    # --------------------------------------------------------------------- #
    cache.invalidate(3)
    assert 3 not in cache
    assert len(cache) == 2

    # --------------------------------------------------------------------- #
    # Test 4: Edge cases – zero/negative max_size handling
    # --------------------------------------------------------------------- #
    try:
        AOTCache[int, int](max_size=0, loader=lambda x: x)
        assert False, "Expected ValueError for max_size=0"
    except ValueError:
        pass

    # --------------------------------------------------------------------- #
    # Test 5: Loader exception propagation
    # --------------------------------------------------------------------- #
    def bad_loader(x: int) -> int:
        raise RuntimeError("forced failure")

    cache = AOTCache[int, int](max_size=2, loader=bad_loader)
    try:
        cache.get(5)
        assert False, "Expected RuntimeError from loader"
    except RuntimeError as e:
        assert str(e) == "forced failure"

    # --------------------------------------------------------------------- #
    # Test 6: Verify that prepopulate does not overwrite existing entries
    # --------------------------------------------------------------------- #
    call_counter['calls'] = 0
    cache = AOTCache[int, int](max_size=5, loader=_loader_factory(call_counter))
    cache.set(10, 42)
    cache.prepopulate([10, 11])
    # Loader should be called only for key 11
    assert call_counter['calls'] == 1
    assert cache.get(10) == 42
    assert cache.get(11) == 39916800  # 11!

    # --------------------------------------------------------------------- #
    # All tests passed
    # --------------------------------------------------------------------- #
    print("All unit tests passed.")


if __name__ == '__main__':
    # Run the verification suite
    run_unit_tests()
    # Demonstration of typical usage
    demo_cache = AOTCache[int, int](max_size=4, loader=lambda n: n * n)
    demo_cache.prepopulate([1, 2, 3])
    print("Demo cache contents after prepopulate:", [demo_cache.get(k) for k in [1, 2, 3]])
    print("Accessing new key 4 (triggers compute):", demo_cache.get(4))
    print("Cache size after accesses:", len(demo_cache))
    # Show LRU eviction by adding a fifth key
    demo_cache.get(5)
    print("Cache keys after eviction (should be 2,3,4,5):", [k for k in demo_cache._store.keys()])
