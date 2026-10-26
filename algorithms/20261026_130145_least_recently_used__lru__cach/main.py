import unittest

from engine import LRUCache


class TestLRUCache(unittest.TestCase):
    def test_basic_operations(self):
        cache = LRUCache[int, int](capacity=2)

        # Cache miss returns None
        self.assertIsNone(cache.get(1))

        # Insert two items
        cache.put(1, 1)
        cache.put(2, 2)
        self.assertEqual(cache.get(1), 1)  # hits key 1, makes it MRU
        self.assertEqual(cache.get(2), 2)

        # Insert third item, should evict key 1 (least recently used)
        cache.put(3, 3)
        self.assertIsNone(cache.get(1))
        self.assertEqual(cache.get(2), 2)
        self.assertEqual(cache.get(3), 3)

    def test_update_existing_key(self):
        cache = LRUCache[str, str](capacity=2)
        cache.put("a", "alpha")
        cache.put("b", "beta")
        cache.put("a", "ALPHA")  # update value and move to MRU
        self.assertEqual(cache.get("a"), "ALPHA")
        # Adding new key should evict "b" because "a" is MRU
        cache.put("c", "gamma")
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), "gamma")

    def test_capacity_zero(self):
        cache = LRUCache[int, int](capacity=0)
        cache.put(1, 1)
        self.assertIsNone(cache.get(1))
        self.assertEqual(len(cache), 0)

    def test_ordering(self):
        cache = LRUCache[int, int](capacity=3)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.put(3, 3)
        self.assertEqual(cache.keys(), [3, 2, 1])  # 3 is most recent

        cache.get(1)  # access 1 -> becomes MRU
        self.assertEqual(cache.keys(), [1, 3, 2])

        cache.put(4, 4)  # evicts LRU (2)
        self.assertIsNone(cache.get(2))
        self.assertEqual(cache.keys(), [4, 1, 3])

    def test_len_and_contains(self):
        cache = LRUCache[int, str](capacity=2)
        self.assertFalse(1 in cache)
        cache.put(1, "one")
        self.assertTrue(1 in cache)
        self.assertEqual(len(cache), 1)
        cache.put(2, "two")
        self.assertEqual(len(cache), 2)
        cache.put(3, "three")  # evicts key 1
        self.assertFalse(1 in cache)
        self.assertTrue(2 in cache)
        self.assertTrue(3 in cache)
        self.assertEqual(len(cache), 2)


if __name__ == '__main__':
    # Run unit tests with higher verbosity for clarity.
    unittest.main(verbosity=2)
