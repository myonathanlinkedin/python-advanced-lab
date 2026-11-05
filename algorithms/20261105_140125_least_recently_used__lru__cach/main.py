import unittest
from engine import LRUCache


class TestLRUCache(unittest.TestCase):
    def test_basic_operations(self):
        cache = LRUCache[int, str](2)
        cache.put(1, "one")
        cache.put(2, "two")
        self.assertEqual(cache.get(1), "one")   # 1 becomes MRU
        cache.put(3, "three")                  # evicts key 2
        self.assertIsNone(cache.get(2))
        self.assertEqual(cache.get(3), "three")
        self.assertEqual(cache.get(1), "one")
        cache.put(4, "four")                    # evicts key 3
        self.assertIsNone(cache.get(3))
        self.assertEqual(cache.get(4), "four")
        self.assertEqual(cache.get(1), "one")

    def test_update_existing(self):
        cache = LRUCache[int, int](2)
        cache.put(1, 1)
        cache.put(1, 10)          # update value, keep MRU status
        self.assertEqual(cache.get(1), 10)
        cache.put(2, 2)
        cache.put(3, 3)           # evicts key 2 (1 is MRU)
        self.assertIsNone(cache.get(2))
        self.assertEqual(cache.get(1), 10)
        self.assertEqual(cache.get(3), 3)

    def test_capacity_zero(self):
        cache = LRUCache[int, int](0)
        cache.put(1, 1)
        self.assertIsNone(cache.get(1))
        cache.put(2, 2)
        self.assertIsNone(cache.get(2))

    def test_large_sequence(self):
        cap = 5
        cache = LRUCache[int, int](cap)
        for i in range(10):
            cache.put(i, i * i)
        # Keys 0‑4 should have been evicted.
        for i in range(5):
            self.assertIsNone(cache.get(i))
        for i in range(5, 10):
            self.assertEqual(cache.get(i), i * i)

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            LRUCache[int, int](-1)


if __name__ == '__main__':
    unittest.main()
