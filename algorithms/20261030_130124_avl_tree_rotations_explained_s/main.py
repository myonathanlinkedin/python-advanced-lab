import unittest
from core import AVLTree

class TestAVLTree(unittest.TestCase):
    def setUp(self):
        self.tree = AVLTree()

    def test_insert_and_inorder(self):
        values = [10, 20, 30, 40, 50, 25]
        for v in values:
            self.tree.insert(v)
        expected = sorted(values)
        self.assertEqual(list(self.tree), expected)

    def test_balancing_after_insert(self):
        # Insert in ascending order to trigger rotations
        for v in range(1, 8):
            self.tree.insert(v)
        # After balancing, root should be 4
        self.assertEqual(self.tree.root.key, 4)

    def test_delete_leaf(self):
        for v in [30, 20, 40]:
            self.tree.insert(v)
        self.tree.delete(20)
        self.assertNotIn(20, self.tree)
        self.assertEqual(list(self.tree), [30, 40])

    def test_delete_node_with_one_child(self):
        for v in [30, 20, 40, 10]:
            self.tree.insert(v)
        self.tree.delete(20)
        self.assertNotIn(20, self.tree)
        self.assertEqual(list(self.tree), [10, 30, 40])

    def test_delete_node_with_two_children(self):
        for v in [50, 30, 70, 20, 40, 60, 80]:
            self.tree.insert(v)
        self.tree.delete(50)
        self.assertNotIn(50, self.tree)
        self.assertEqual(list(self.tree), [20, 30, 40, 60, 70, 80])

    def test_duplicate_insertion(self):
        self.tree.insert(15)
        self.tree.insert(15)
        self.assertEqual(len(self.tree), 1)

    def test_empty_tree(self):
        self.assertEqual(len(self.tree), 0)
        self.assertFalse(10 in self.tree)

    def test_height_and_balance(self):
        # Build a balanced tree
        for v in [10, 20, 30, 40, 50]:
            self.tree.insert(v)
        # Root should be 30
        self.assertEqual(self.tree.root.key, 30)
        # Height of root should be 3
        self.assertEqual(self.tree.root.height, 3)

if __name__ == "__main__":
    # Demo
    demo_tree = AVLTree()
    for val in [15, 10, 20, 8, 12, 17, 25]:
        demo_tree.insert(val)
    print("In-order traversal of AVL tree:", list(demo_tree))
    unittest.main(exit=False)
