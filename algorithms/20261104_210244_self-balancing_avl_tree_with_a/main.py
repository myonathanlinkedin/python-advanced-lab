import unittest
import random
import time
from typing import List, Tuple
from core import AVLTree

class TestAVLTree(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = AVLTree[int, str]()

    def test_insert_and_find(self) -> None:
        pairs = [(5, 'a'), (3, 'b'), (8, 'c'), (1, 'd'), (4, 'e')]
        for k, v in pairs:
            self.tree.insert(k, v)
        for k, v in pairs:
            self.assertEqual(self.tree.find(k), v)
        self.assertIsNone(self.tree.find(999))

    def test_inorder_sorted(self) -> None:
        keys = list(range(20))
        random.shuffle(keys)
        for k in keys:
            self.tree.insert(k, str(k))
        inorder = self.tree.inorder()
        extracted_keys = [k for k, _ in inorder]
        self.assertEqual(extracted_keys, sorted(keys))
        self.assertTrue(self.tree.is_balanced())

    def test_duplicate_key_updates(self) -> None:
        self.tree.insert(10, 'first')
        self.assertEqual(self.tree.find(10), 'first')
        self.tree.insert(10, 'second')
        self.assertEqual(self.tree.find(10), 'second')
        self.assertTrue(self.tree.is_balanced())

    def test_delete_leaf(self) -> None:
        for k in [20, 10, 30]:
            self.tree.insert(k, str(k))
        self.tree.delete(10)  # leaf
        self.assertIsNone(self.tree.find(10))
        self.assertTrue(self.tree.is_balanced())

    def test_delete_one_child(self) -> None:
        for k in [20, 10, 5]:
            self.tree.insert(k, str(k))
        self.tree.delete(10)  # node with left child only
        self.assertIsNone(self.tree.find(10))
        self.assertEqual(self.tree.find(5), '5')
        self.assertTrue(self.tree.is_balanced())

    def test_delete_two_children(self) -> None:
        for k in [40, 20, 60, 10, 30, 50, 70]:
            self.tree.insert(k, str(k))
        self.tree.delete(20)  # node with two children
        self.assertIsNone(self.tree.find(20))
        self.assertTrue(self.tree.is_balanced())
        # Ensure remaining keys are present
        for k in [10, 30, 40, 50, 60, 70]:
            self.assertIsNotNone(self.tree.find(k))

    def test_random_operations(self) -> None:
        keys = list(range(100))
        random.shuffle(keys)
        for k in keys:
            self.tree.insert(k, str(k))
        self.assertTrue(self.tree.is_balanced())

        # Random deletions
        to_delete = random.sample(keys, 30)
        for k in to_delete:
            self.tree.delete(k)
        self.assertTrue(self.tree.is_balanced())

        # Verify remaining keys
        remaining = set(keys) - set(to_delete)
        for k in remaining:
            self.assertEqual(self.tree.find(k), str(k))
        for k in to_delete:
            self.assertIsNone(self.tree.find(k))

    def test_performance_small(self) -> None:
        start = time.perf_counter()
        for i in range(1000):
            self.tree.insert(i, str(i))
        duration = time.perf_counter() - start
        # Simple sanity check: should be fast (<0.5s on typical hardware)
        self.assertLess(duration, 0.5)

if __name__ == '__main__':
    unittest.main()
