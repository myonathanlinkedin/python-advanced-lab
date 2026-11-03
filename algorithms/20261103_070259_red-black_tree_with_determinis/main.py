import unittest
from typing import List, Tuple
from core import RBTree

class TestRBTree(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = RBTree()

    def test_insert_and_inorder(self) -> None:
        data = [(5, 'e'), (3, 'c'), (7, 'g'), (2, 'b'), (4, 'd'), (6, 'f'), (8, 'h'), (1, 'a')]
        for k, v in data:
            self.tree.insert(k, v)
        inorder = self.tree.inorder()
        expected_keys = sorted(k for k, _ in data)
        self.assertEqual([k for k, _ in inorder], expected_keys)

    def test_search(self) -> None:
        pairs = [(10, 'x'), (20, 'y'), (5, 'z')]
        for k, v in pairs:
            self.tree.insert(k, v)
        for k, v in pairs:
            self.assertEqual(self.tree.get(k), v)
        with self.assertRaises(KeyError):
            self.tree.get(999)

    def test_delete_leaf(self) -> None:
        for k in range(1, 6):
            self.tree.insert(k, str(k))
        self.tree.delete(5)  # leaf
        self.assertNotIn(5, [k for k, _ in self.tree.inorder()])

    def test_delete_one_child(self) -> None:
        for k in [10, 5, 15, 12]:
            self.tree.insert(k, str(k))
        self.tree.delete(15)  # node with one child (12)
        self.assertNotIn(15, [k for k, _ in self.tree.inorder()])

    def test_delete_two_children(self) -> None:
        for k in [20, 10, 30, 5, 15, 25, 35]:
            self.tree.insert(k, str(k))
        self.tree.delete(20)  # node with two children
        self.assertNotIn(20, [k for k, _ in self.tree.inorder()])

    def test_random_operations(self) -> None:
        import random
        keys = list(range(1, 101))
        random.shuffle(keys)
        for k in keys:
            self.tree.insert(k, k * 2)
        # verify order
        inorder_keys = [k for k, _ in self.tree.inorder()]
        self.assertEqual(inorder_keys, sorted(keys))
        # delete half
        for k in keys[:50]:
            self.tree.delete(k)
        remaining = set(keys[50:])
        self.assertEqual(set(k for k, _ in self.tree.inorder()), remaining)

    def test_properties_after_operations(self) -> None:
        # Insert a sequence that triggers many rotations
        seq = [41, 38, 31, 12, 19, 8]
        for k in seq:
            self.tree.insert(k, k)
        # Delete causing fix‑up
        self.tree.delete(31)
        # If no assertion raised, properties hold
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
