import unittest
from typing import List, Tuple

from core import Treap


class TestTreap(unittest.TestCase):
    def setUp(self) -> None:
        # Fixed seed for deterministic priority generation
        self.t = Treap(seed=42)

    def test_insert_and_find(self) -> None:
        data = [(5, 'a'), (3, 'b'), (8, 'c'), (1, 'd'), (4, 'e')]
        for k, v in data:
            self.t.insert(k, v)
        for k, v in data:
            self.assertEqual(self.t.find(k), v)
        self.assertIsNone(self.t.find(999))

    def test_duplicate_key_updates_value(self) -> None:
        self.t.insert(10, 'first')
        self.assertEqual(self.t.find(10), 'first')
        self.t.insert(10, 'second')
        self.assertEqual(self.t.find(10), 'second')
        # Ensure structure still valid
        self.assertTrue(self.t.validate())

    def test_erase(self) -> None:
        keys = [20, 5, 15, 25, 2, 7]
        for k in keys:
            self.t.insert(k, str(k))
        self.t.erase(15)
        self.assertIsNone(self.t.find(15))
        self.t.erase(100)  # non‑existent key – should be no‑op
        # Remaining keys must be present
        for k in [20, 5, 25, 2, 7]:
            self.assertEqual(self.t.find(k), str(k))
        self.assertTrue(self.t.validate())

    def test_inorder_traversal(self) -> None:
        pairs = [(50, 'a'), (30, 'b'), (70, 'c'), (20, 'd'), (40, 'e'), (60, 'f'), (80, 'g')]
        for k, v in pairs:
            self.t.insert(k, v)
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        self.assertEqual(self.t.inorder(), sorted_pairs)

    def test_randomized_operations_preserve_invariants(self) -> None:
        # Insert a large number of random keys
        import random
        rnd = random.Random(123)
        keys = rnd.sample(range(1, 1000), 200)
        for k in keys:
            self.t.insert(k, k * 2)
            self.assertTrue(self.t.validate())
        # Random deletions
        del_keys = rnd.sample(keys, 50)
        for k in del_keys:
            self.t.erase(k)
            self.assertTrue(self.t.validate())
        # Verify remaining keys
        remaining = set(keys) - set(del_keys)
        for k in remaining:
            self.assertEqual(self.t.find(k), k * 2)

    def test_edge_cases(self) -> None:
        # Empty treap
        self.assertIsNone(self.t.find(1))
        self.assertTrue(self.t.validate())
        # Single element
        self.t.insert(0, 'zero')
        self.assertEqual(self.t.find(0), 'zero')
        self.assertTrue(self.t.validate())
        self.t.erase(0)
        self.assertIsNone(self.t.find(0))
        self.assertTrue(self.t.validate())


if __name__ == '__main__':
    unittest.main(verbosity=2)
