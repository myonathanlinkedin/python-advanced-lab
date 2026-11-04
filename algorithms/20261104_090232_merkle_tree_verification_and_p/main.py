"""
Unit tests and demo for the MerkleTree implementation.
"""

import unittest
from typing import List

from core import MerkleTree, Hash


class TestMerkleTree(unittest.TestCase):
    def test_single_leaf(self):
        leaves = [b"leaf1"]
        tree = MerkleTree(leaves)
        root = tree.get_root()
        proof = tree.get_proof(0)
        self.assertTrue(MerkleTree.verify_proof(b"leaf1", proof, root))
        self.assertFalse(MerkleTree.verify_proof(b"leaf2", proof, root))

    def test_multiple_leaves(self):
        leaves = [b"leaf1", b"leaf2", b"leaf3", b"leaf4"]
        tree = MerkleTree(leaves)
        root = tree.get_root()
        for i, leaf in enumerate(leaves):
            proof = tree.get_proof(i)
            self.assertTrue(MerkleTree.verify_proof(leaf, proof, root))
            self.assertFalse(MerkleTree.verify_proof(b"tampered", proof, root))

    def test_odd_leaves(self):
        leaves = [b"a", b"b", b"c"]
        tree = MerkleTree(leaves)
        root = tree.get_root()
        for i, leaf in enumerate(leaves):
            proof = tree.get_proof(i)
            self.assertTrue(MerkleTree.verify_proof(leaf, proof, root))

    def test_empty_leaves(self):
        with self.assertRaises(ValueError):
            MerkleTree([])

    def test_proof_length(self):
        leaves = [b"1", b"2", b"3", b"4", b"5"]
        tree = MerkleTree(leaves)
        expected_len = len(tree._levels) - 1
        for i in range(len(leaves)):
            proof = tree.get_proof(i)
            self.assertEqual(len(proof), expected_len)

    def test_duplicate_last_node(self):
        leaves = [b"1", b"2", b"3"]
        tree = MerkleTree(leaves)
        root = tree.get_root()
        proof = tree.get_proof(2)
        self.assertTrue(MerkleTree.verify_proof(b"3", proof, root))

    def test_verify_false_on_wrong_root(self):
        leaves = [b"1", b"2"]
        tree = MerkleTree(leaves)
        root = tree.get_root()
        proof = tree.get_proof(0)
        wrong_root = b"\x00" * 32
        self.assertFalse(MerkleTree.verify_proof(b"1", proof, wrong_root))

    def test_proof_index_out_of_range(self):
        leaves = [b"leaf1", b"leaf2"]
        tree = MerkleTree(leaves)
        with self.assertRaises(IndexError):
            tree.get_proof(2)

    def test_str_input(self):
        leaves = ["a", "b", "c"]
        tree = MerkleTree(leaves)
        root = tree.get_root()
        for i, leaf in enumerate(leaves):
            proof = tree.get_proof(i)
            self.assertTrue(MerkleTree.verify_proof(leaf.encode(), proof, root))

    def test_root_consistency(self):
        leaves = [b"data1", b"data2", b"data3", b"data4"]
        tree1 = MerkleTree(leaves)
        tree2 = MerkleTree(leaves)
        self.assertEqual(tree1.get_root(), tree2.get_root())


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demo: build a tree and verify a proof
    demo_leaves = [b"apple", b"banana", b"cherry", b"date"]
    tree = MerkleTree(demo_leaves)
    print(f"Merkle Root: {tree.get_root().hex()}")
    for idx, leaf in enumerate(demo_leaves):
        proof = tree.get_proof(idx)
        valid = MerkleTree.verify_proof(leaf, proof, tree.get_root())
        print(f"Leaf {idx} ({leaf.decode()}): proof valid? {valid}")
        # Show proof details
        print("Proof:")
        for sibling_hash, is_left in proof:
            side = "left" if is_left else "right"
            print(f"  {side} sibling: {sibling_hash.hex()}")
        print()
