import unittest
from typing import List, Tuple

# Import the core implementation
from core import SkipList


class TestSkipList(unittest.TestCase):
    def setUp(self) -> None:
        # Fixed seed for deterministic level generation
        self.sl = SkipList(max_level=8, p=0.5, seed=42)

    def test_insert_and_search(self) -> None:
        pairs = [(5, "five"), (1, "one"), (3, "three"), (10, "ten")]
        for k, v in pairs:
            self.sl.insert(k, v)

        for k, v in pairs:
            self.assertEqual(self.sl.search(k), v)

        # Search for missing key
        self.assertIsNone(self.sl.search(99))

    def test_update_existing_key(self) -> None:
        self.sl.insert(2, "two")
        self.assertEqual(self.sl.search(2), "two")
        self.sl.insert(2, "deux")
        self.assertEqual(self.sl.search(2), "deux")
        self.assertEqual(len(self.sl), 1)

    def test_order_preservation(self) -> None:
        keys = [7, 3, 9, 1, 5]
        for k in keys:
            self.sl.insert(k, str(k))
        sorted_keys = sorted(keys)
        iterated_keys = [k for k, _ in self.sl]
        self.assertEqual(iterated_keys, sorted_keys)

    def test_delete(self) -> None:
        for k in range(1, 6):
            self.sl.insert(k, str(k))
        self.assertTrue(self.sl.delete(3))
        self.assertIsNone(self.sl.search(3))
        self.assertFalse(self.sl.delete(3))  # already removed
        self.assertEqual(len(self.sl), 4)

        remaining = [k for k, _ in self.sl]
        self.assertEqual(remaining, [1, 2, 4, 5])

    def test_contains(self) -> None:
        self.sl.insert(42, "answer")
        self.assertIn(42, self.sl)
        self.assertNotIn(100, self.sl)

    def test_random_level_distribution(self) -> None:
        # With a fixed seed we can assert the generated levels for a known sequence
        levels: List[int] = []
        for _ in range(10):
            levels.append(self.sl._random_level())
        expected = [0, 1, 0, 0, 2, 0, 1, 0, 0, 0]  # pre‑computed for seed=42, p=0.5, max_level=8
        self.assertEqual(levels, expected)

    def test_dump_levels_structure(self) -> None:
        for k in [10, 20, 30]:
            self.sl.insert(k, str(k))
        levels = self.sl.dump_levels()
        # Level 0 must contain all keys in order
        self.assertEqual(levels[0], [10, 20, 30])
        # Higher levels may be empty depending on random levels; just verify type
        for lvl in levels:
            self.assertIsInstance(lvl, list)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
