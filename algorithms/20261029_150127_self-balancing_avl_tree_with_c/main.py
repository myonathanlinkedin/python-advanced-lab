import unittest
from core import AVLTree

class TestAVLTree(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = AVLTree()

    def test_insert_and_search(self) -> None:
        self.tree.insert(10, 'a')
        self.tree.insert(20, 'b')
        self.tree.insert(5, 'c')
        self.assertEqual(self.tree.search(10), 'a')
        self.assertEqual(self.tree.search(20), 'b')
        self.assertEqual(self.tree.search(5), 'c')
        self.assertIsNone(self.tree.search(99))

    def test_inorder(self) -> None:
        keys = [15, 10, 20, 8, 12, 17, 25]
        for i, k in enumerate(keys):
            self.tree.insert(k, f'v{i}')
        inorder = self.tree.inorder()
        self.assertEqual([k for k, _, _ in inorder], sorted(keys))

    def test_balancing(self) -> None:
        for i in range(1, 16):
            self.tree.insert(i, str(i))
        def height(node):
            if not node:
                return 0
            return 1 + max(height(node.left), height(node.right))
        h = height(self.tree.root)
        self.assertLessEqual(h, 5)

    def test_delete(self) -> None:
        for k in [10, 20, 5, 15, 25]:
            self.tree.insert(k, str(k))
        self.tree.delete(20)
        self.assertIsNone(self.tree.search(20))
        inorder = self.tree.inorder()
        self.assertEqual([k for k, _, _ in inorder], [5, 10, 15, 25])

    def test_timestamp_order(self) -> None:
        for i in range(5):
            self.tree.insert(i, f'v{i}')
        timestamps = [ts for _, _, ts in self.tree.inorder()]
        self.assertEqual(timestamps, sorted(timestamps))

if __name__ == '__main__':
    unittest.main(exit=False)
    # Demo
    demo_tree = AVLTree()
    for k in [30, 20, 40, 10, 25, 35, 50]:
        demo_tree.insert(k, f'val{k}')
    print('In-order traversal:', demo_tree.inorder())
