import unittest
from core import Treap


class TestTreap(unittest.TestCase):
    def setUp(self):
        self.treap = Treap(seed=42)

    def test_empty(self):
        self.assertEqual(self.treap.size(), 0)
        self.assertFalse(self.treap.search(10))
        self.assertEqual(self.treap.inorder(), [])

    def test_insert_and_search(self):
        keys = [5, 3, 7, 2, 4, 6, 8]
        for k in keys:
            self.treap.insert(k)
        self.assertEqual(self.treap.size(), len(keys))
        for k in keys:
            self.assertTrue(self.treap.search(k))
        self.assertFalse(self.treap.search(10))

    def test_inorder_sorted(self):
        keys = [10, 5, 15, 3, 7, 13, 17]
        for k in keys:
            self.treap.insert(k)
        self.assertEqual(self.treap.inorder(), sorted(keys))

    def test_delete_leaf(self):
        keys = [5, 3, 7]
        for k in keys:
            self.treap.insert(k)
        self.treap.delete(3)
        self.assertFalse(self.treap.search(3))
        self.assertEqual(self.treap.size(), 2)
        self.assertEqual(self.treap.inorder(), [5, 7])

    def test_delete_internal(self):
        keys = [5, 3, 7, 2, 4]
        for k in keys:
            self.treap.insert(k)
        self.treap.delete(5)
        self.assertFalse(self.treap.search(5))
        self.assertEqual(self.treap.size(), 4)
        self.assertEqual(self.treap.inorder(), [2, 3, 4, 7])

    def test_delete_root(self):
        keys = [5, 3, 7]
        for k in keys:
            self.treap.insert(k)
        self.treap.delete(5)
        self.assertFalse(self.treap.search(5))
        self.assertEqual(self.treap.size(), 2)
        self.assertEqual(self.treap.inorder(), [3, 7])

    def test_split(self):
        keys = [5, 3, 7, 2, 4, 6, 8]
        for k in keys:
            self.treap.insert(k)
        left, right = self.treap.split(6)
        self.assertEqual(left.inorder(), [2, 3, 4, 5])
        self.assertEqual(right.inorder(), [6, 7, 8])

    def test_merge(self):
        left = Treap(seed=1)
        right = Treap(seed=2)
        for k in [1, 2, 3]:
            left.insert(k)
        for k in [4, 5, 6]:
            right.insert(k)
        merged = Treap.merge(left, right)
        self.assertEqual(merged.inorder(), [1, 2, 3, 4, 5, 6])

    def test_no_duplicates(self):
        self.treap.insert(5)
        with self.assertRaises(ValueError):
            self.treap.insert(5)

    def test_delete_nonexistent(self):
        with self.assertRaises(KeyError):
            self.treap.delete(99)

    def test_heap_property(self):
        # After random insertions, verify heap property
        keys = [i for i in range(20)]
        for k in keys:
            self.treap.insert(k)
        def check_heap(node):
            if node is None:
                return True
            left_ok = node.left is None or node.priority >= node.left.priority
            right_ok = node.right is None or node.priority >= node.right.priority
            return left_ok and right_ok and check_heap(node.left) and check_heap(node.right)
        self.assertTrue(check_heap(self.treap.root))


if __name__ == '__main__':
    unittest.main()
    # Demo
    demo = Treap(seed=123)
    for val in [10, 20, 5, 15, 25]:
        demo.insert(val)
    print("Inorder traversal:", demo.inorder())
    demo.delete(15)
    print("After deleting 15:", demo.inorder())
    left, right = demo.split(18)
    print("Left split:", left.inorder())
    print("Right split:", right.inorder())
    merged = Treap.merge(left, right)
    print("Merged inorder:", merged.inorder())
