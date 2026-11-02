"""
main.py
---------

Unit‑test suite for the LRUCache implementation.
Run the module directly to execute tests and a brief demonstration.
"""

import unittest
import timeit
from typing import Any

from core import LRUCache


class TestLRUCacheBasic(unittest.TestCase):
    def test_put_and_get(self) -> None:
        cache = LRUCache(2)
        cache.put(1, "a")
        cache.put(2, "b")
        self.assertEqual(cache.get(1), "a")
        self.assertEqual(cache.get(2), "b")
        self.assertEqual(len(cache), 2)

    def test_update_existing_key(self) -> None:
        cache = LRUCache(2)
        cache.put("x", 10)
        cache.put("x", 20)  # update
        self.assertEqual(cache.get("x"), 20)
        self.assertEqual(len(cache), 1)

    def test_eviction_order(self) -> None:
        cache = LRUCache(2)
        cache.put(1, "one")
        cache.put(2, "two")
        cache.get(1)          # 1 becomes most recent
        cache.put(3, "three")  # evicts key 2
        self.assertIsNone(cache.get(2))
        self.assertEqual(cache.get(1), "one")
        self.assertEqual(cache.get(3), "three")
        self.assertEqual(len(cache), 2)

    def test_capacity_zero(self) -> None:
        cache = LRUCache(0)
        cache.put(1, "a")
        self.assertIsNone(cache.get(1))
        self.assertEqual(len(cache), 0)

    def test_contains(self) -> None:
        cache = LRUCache(1)
        cache.put("k", 100)
        self.assertIn("k", cache)
        cache.put("x", 200)  # evicts "k"
        self.assertNotIn("k", cache)
        self.assertIn("x", cache)


class TestLRUCacheStress(unittest.TestCase):
    def test_large_sequence(self) -> None:
        capacity = 100
        ops = 10_000
        cache = LRUCache(capacity)
        for i in range(ops):
            cache.put(i, i * 2)
            # Access some earlier keys to keep them alive
            if i % 50 == 0:
                _ = cache.get(i - 30)
        # After all puts, only the last `capacity` keys should remain
        for i in range(ops - capacity, ops):
            self.assertEqual(cache.get(i), i * 2)
        self.assertEqual(len(cache), capacity)

    def test_time_complexity(self) -> None:
        """A very light benchmark to ensure O(1) behaviour."""
        cache = LRUCache(1000)
        setup = "from __main__ import cache"
        stmt = "cache.put(1, 'a'); cache.get(1)"
        # Run 10000 iterations; should complete quickly (<0.5s)
        duration = timeit.timeit(stmt, setup=setup, number=10000)
        self.assertLess(duration, 0.5)


def demo() -> None:
    """Simple interactive demonstration."""
    print("=== LRU Cache Demo ===")
    cache = LRUCache(3)
    actions = [
        ("put", 1, "A"),
        ("put", 2, "B"),
        ("put", 3, "C"),
        ("get", 2, None),
        ("put", 4, "D"),  # should evict key 1
        ("get", 1, None),
        ("get", 3, None),
        ("get", 4, None),
    ]
    for act, key, val in actions:
        if act == "put":
            cache.put(key, val)
            print(f"put({key}, {val}) -> {cache}")
        else:  # get
            result = cache.get(key)
            print(f"get({key}) -> {result} | {cache}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    demo()
