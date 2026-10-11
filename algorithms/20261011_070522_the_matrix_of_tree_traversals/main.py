import unittest
from typing import List, Any

from core import (
    TreeNode,
    recursive_preorder,
    recursive_inorder,
    recursive_postorder,
    iterative_preorder,
    iterative_inorder,
    iterative_postorder,
    traversal_matrix,
)


def build_sample_tree() -> TreeNode:
    """
    Build the following binary tree used in many tests:

            1
          /   \
         2     3
        / \   / \
       4   5 6   7
    """
    n4 = TreeNode(4)
    n5 = TreeNode(5)
    n6 = TreeNode(6)
    n7 = TreeNode(7)
    n2 = TreeNode(2, n4, n5)
    n3 = TreeNode(3, n6, n7)
    root = TreeNode(1, n2, n3)
    return root


class TestTreeTraversals(unittest.TestCase):
    def test_empty_tree(self):
        self.assertEqual(recursive_preorder(None), [])
        self.assertEqual(iterative_preorder(None), [])
        self.assertEqual(recursive_inorder(None), [])
        self.assertEqual(iterative_inorder(None), [])
        self.assertEqual(recursive_postorder(None), [])
        self.assertEqual(iterative_postorder(None), [])
        matrix = traversal_matrix(None)
        for name in ("preorder", "inorder", "postorder"):
            self.assertEqual(matrix[name][0], [])
            self.assertEqual(matrix[name][1], [])

    def test_single_node(self):
        root = TreeNode("root")
        expected = ["root"]
        self.assertEqual(recursive_preorder(root), expected)
        self.assertEqual(iterative_preorder(root), expected)
        self.assertEqual(recursive_inorder(root), expected)
        self.assertEqual(iterative_inorder(root), expected)
        self.assertEqual(recursive_postorder(root), expected)
        self.assertEqual(iterative_postorder(root), expected)
        matrix = traversal_matrix(root)
        for name in ("preorder", "inorder", "postorder"):
            self.assertEqual(matrix[name][0], expected)
            self.assertEqual(matrix[name][1], expected)

    def test_balanced_tree(self):
        root = build_sample_tree()
        self.assertEqual(recursive_preorder(root), [1, 2, 4, 5, 3, 6, 7])
        self.assertEqual(iterative_preorder(root), [1, 2, 4, 5, 3, 6, 7])

        self.assertEqual(recursive_inorder(root), [4, 2, 5, 1, 6, 3, 7])
        self.assertEqual(iterative_inorder(root), [4, 2, 5, 1, 6, 3, 7])

        self.assertEqual(recursive_postorder(root), [4, 5, 2, 6, 7, 3, 1])
        self.assertEqual(iterative_postorder(root), [4, 5, 2, 6, 7, 3, 1])

        matrix = traversal_matrix(root)
        for name in ("preorder", "inorder", "postorder"):
            rec, itr = matrix[name]
            self.assertEqual(rec, itr)

    def test_unbalanced_tree(self):
        """
        Build an unbalanced tree:

                10
               /
              5
               \
                7
               /
              6
        """
        n6 = TreeNode(6)
        n7 = TreeNode(7, n6, None)
        n5 = TreeNode(5, None, n7)
        root = TreeNode(10, n5, None)

        self.assertEqual(recursive_preorder(root), [10, 5, 7, 6])
        self.assertEqual(iterative_preorder(root), [10, 5, 7, 6])

        self.assertEqual(recursive_inorder(root), [5, 6, 7, 10])
        self.assertEqual(iterative_inorder(root), [5, 6, 7, 10])

        self.assertEqual(recursive_postorder(root), [6, 7, 5, 10])
        self.assertEqual(iterative_postorder(root), [6, 7, 5, 10])

        matrix = traversal_matrix(root)
        for name in ("preorder", "inorder", "postorder"):
            self.assertEqual(matrix[name][0], matrix[name][1])

    def test_random_structure_consistency(self):
        """
        Randomly generate a small tree and verify recursive vs iterative match.
        """
        # Manual construction of a more irregular tree
        #        a
        #       / \
        #      b   c
        #       \   \
        #        d   e
        #       /   /
        #      f   g
        f = TreeNode('f')
        d = TreeNode('d', f, None)
        b = TreeNode('b', None, d)
        g = TreeNode('g')
        e = TreeNode('e', g, None)
        c = TreeNode('c', None, e)
        a = TreeNode('a', b, c)

        self.assertEqual(recursive_preorder(a), ['a', 'b', 'd', 'f', 'c', 'e', 'g'])
        self.assertEqual(iterative_preorder(a), ['a', 'b', 'd', 'f', 'c', 'e', 'g'])

        self.assertEqual(recursive_inorder(a), ['b', 'f', 'd', 'a', 'c', 'g', 'e'])
        self.assertEqual(iterative_inorder(a), ['b', 'f', 'd', 'a', 'c', 'g', 'e'])

        self.assertEqual(recursive_postorder(a), ['f', 'd', 'b', 'g', 'e', 'c', 'a'])
        self.assertEqual(iterative_postorder(a), ['f', 'd', 'b', 'g', 'e', 'c', 'a'])

        matrix = traversal_matrix(a)
        for name in ("preorder", "inorder", "postorder"):
            self.assertEqual(matrix[name][0], matrix[name][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
