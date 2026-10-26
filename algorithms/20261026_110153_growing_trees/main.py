"""Unit tests, benchmarks, and entrypoint for GrowingTree."""
from __future__ import annotations

import time
import unittest
from typing import List

from core import GrowingTree


class TestGrowingTreeConstruction(unittest.TestCase):
    """Verify construction invariants and determinism."""

    def test_single_node(self) -> None:
        t = GrowingTree(1, seed=42)
        self.assertEqual(t.n, 1)
        self.assertEqual(t.parent[1], 0)
        self.assertEqual(t.depth[1], 0)
        self.assertEqual(t.children[1], [])
        self.assertTrue(t.is_valid())

    def test_two_nodes(self) -> None:
        t = GrowingTree(2, seed=0)
        self.assertEqual(t.parent[2], 1)
        self.assertEqual(t.depth[2], 1)
        self.assertEqual(t.children[1], [2])
        self.assertTrue(t.is_valid())

    def test_determinism_with_seed(self) -> None:
        t1 = GrowingTree(100, seed=12345)
        t2 = GrowingTree(100, seed=12345)
        self.assertEqual(t1.parent, t2.parent)
        self.assertEqual(t1.children, t2.children)
        self.assertEqual(t1.depth, t2.depth)

    def test_different_seeds_differ(self) -> None:
        t1 = GrowingTree(50, seed=1)
        t2 = GrowingTree(50, seed=2)
        # With high probability, different seeds yield different structures.
        self.assertNotEqual(t1.parent, t2.parent)

    def test_invalid_n_raises(self) -> None:
        with self.assertRaises(ValueError):
            GrowingTree(0)
        with self.assertRaises(ValueError):
            GrowingTree(-5)

    def test_validity_for_many_sizes(self) -> None:
        for n in [1, 2, 3, 5, 10, 50, 100, 500]:
            t = GrowingTree(n, seed=n)
            self.assertTrue(t.is_valid(), f"Invalid tree for n={n}")


class TestGrowingTreeQueries(unittest.TestCase):
    """Verify query methods: subtree_size, path_to_root, leaves, degree."""

    def setUp(self) -> None:
        self.t = GrowingTree(20, seed=99)

    def test_subtree_size_root(self) -> None:
        self.assertEqual(self.t.subtree_size(1), 20)

    def test_subtree_size_leaf(self) -> None:
        for leaf in self.t.leaves():
            self.assertEqual(self.t.subtree_size(leaf), 1)

    def test_subtree_size_sum_consistency(self) -> None:
        # Sum of subtree sizes of root's children + 1 (root) == n.
        total = 1
        for c in self.t.children[1]:
            total += self.t.subtree_size(c)
        self.assertEqual(total, self.t.n)

    def test_path_to_root_root(self) -> None:
        self.assertEqual(self.t.path_to_root(1), [1])

    def test_path_to_root_structure(self) -> None:
        for node in range(1, self.t.n + 1):
            path = self.t.path_to_root(node)
            self.assertEqual(path[0], node)
            self.assertEqual(path[-1], 1)
            # Each consecutive pair must be parent-child.
            for i in range(len(path) - 1):
                self.assertEqual(self.t.parent[path[i]], path[i + 1])
            # Path length == depth + 1.
            self.assertEqual(len(path), self.t.depth[node] + 1)

    def test_leaves_count(self) -> None:
        leaves = self.t.leaves()
        for leaf in leaves:
            self.assertEqual(self.t.children[leaf], [])
        # Every non-leaf has at least one child.
        for i in range(1, self.t.n + 1):
            if i not in leaves:
                self.assertGreater(len(self.t.children[i]), 0)

    def test_degree_root(self) -> None:
        self.assertEqual(self.t.degree(1), len(self.t.children[1]))

    def test_degree_non_root(self) -> None:
        for i in range(2, self.t.n + 1):
            expected = len(self.t.children[i]) + 1
            self.assertEqual(self.t.degree(i), expected)

    def test_max_depth(self) -> None:
        expected = max(self.t.depth[1:])
        self.assertEqual(self.t.max_depth(), expected)

    def test_index_error_out_of_range(self) -> None:
        with self.assertRaises(IndexError):
            self.t.subtree_size(0)
        with self.assertRaises(IndexError):
            self.t.subtree_size(21)
        with self.assertRaises(IndexError):
            self.t.path_to_root(0)
        with self.assertRaises(IndexError):
            self.t.degree(0)


class TestGrowingTreeStatistical(unittest.TestCase):
    """Statistical sanity checks on RRT properties."""

    def test_expected_depth_order(self) -> None:
        # For RRT, expected depth of node n is ~ 2 ln n.
        # Verify max_depth is O(log n) for large n (loose bound).
        n = 10000
        t = GrowingTree(n, seed=7)
        md = t.max_depth()
        # Loose upper bound: depth < 10 * log2(n) for n=10000 (~133).
        self.assertLess(md, 10 * 14)  # 10 * log2(10000) ~ 133

    def test_edge_count(self) -> None:
        for n in [1, 2, 10, 100]:
            t = GrowingTree(n, seed=0)
            edges = sum(len(ch) for ch in t.children)
            self.assertEqual(edges, n - 1)


class TestGrowingTreeBenchmark(unittest.TestCase):
    """Lightweight performance sanity checks."""

    def test_construction_time(self) -> None:
        n = 100_000
        start = time.perf_counter()
        t = GrowingTree(n, seed=42)
        elapsed = time.perf_counter() - start
        # Should complete well under 2 seconds on modern hardware.
        self.assertLess(elapsed, 2.0)
        self.assertTrue(t.is_valid())

    def test_subtree_size_large(self) -> None:
        n = 50_000
        t = GrowingTree(n, seed=1)
        start = time.perf_counter()
        size = t.subtree_size(1)
        elapsed = time.perf_counter() - start
        self.assertEqual(size, n)
        self.assertLess(elapsed, 1.0)


def _demo() -> None:
    """Runnable demonstration of GrowingTree."""
    print("=== Growing Tree Demo ===")
    t = GrowingTree(15, seed=2024)
    print(f"n = {t.n}, max_depth = {t.max_depth()}")
    print(f"Root children: {t.children[1]}")
    print(f"Leaves: {t.leaves()}")
    print(f"Path to root (node 15): {t.path_to_root(15)}")
    print(f"Subtree size (node 1): {t.subtree_size(1)}")
    print(f"Valid: {t.is_valid()}")
    print()

    # Show a small tree structure.
    small = GrowingTree(8, seed=42)
    print("Small tree (n=8, seed=42):")
    for i in range(1, small.n + 1):
        indent = "  " * small.depth[i]
        print(f"{indent}Node {i} (depth={small.depth[i]}, children={small.children[i]})")


if __name__ == "__main__":
    _demo()
    unittest.main(verbosity=2)
