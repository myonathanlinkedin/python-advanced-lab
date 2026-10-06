"""Unit tests, simple benchmark and demo for DisjointSetMatrix."""

import random
import time
import unittest
from typing import Tuple

from core import DisjointSetMatrix


class TestDisjointSetMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = DisjointSetMatrix(3, 4)  # 3 rows, 4 cols

    def test_initial_state(self) -> None:
        for r in range(3):
            for c in range(4):
                self.assertEqual(self.ds.find(r, c), (r, c))
                self.assertEqual(self.ds.size(r, c), 1)

    def test_union_and_find(self) -> None:
        self.assertTrue(self.ds.union(0, 0, 0, 1))
        self.assertTrue(self.ds.connected(0, 0, 0, 1))
        root = self.ds.find(0, 0)
        self.assertEqual(self.ds.find(0, 1), root)
        self.assertEqual(self.ds.size(*root), 2)

    def test_idempotent_union(self) -> None:
        self.ds.union(1, 2, 2, 3)
        self.assertFalse(self.ds.union(1, 2, 2, 3))  # already same set

    def test_get_set(self) -> None:
        cells = [(0, 0), (0, 1), (1, 0)]
        for a, b in zip(cells, cells[1:]):
            self.ds.union(*a, *b)
        members = self.ds.get_set(0, 0)
        self.assertCountEqual(members, cells)

    def test_invalid_indices(self) -> None:
        with self.assertRaises(IndexError):
            self.ds.find(-1, 0)
        with self.assertRaises(IndexError):
            self.ds.union(0, 0, 3, 0)  # row 3 out of range

    def test_sets_enumeration(self) -> None:
        self.ds.union(0, 0, 0, 1)
        self.ds.union(2, 2, 2, 3)
        all_sets = self.ds.sets()
        # Expect three sets: one of size 2, another of size 2, and the rest singletons
        sizes = sorted(len(s) for s in all_sets)
        self.assertEqual(sizes[-2:], [2, 2])  # two largest sets have size 2


def benchmark(rows: int = 200, cols: int = 200, ops: int = 100_000) -> Tuple[float, float]:
    """Run a simple benchmark: random unions on a matrix.

    Returns (union_time, find_time) in seconds.
    """
    ds = DisjointSetMatrix(rows, cols)
    rand = random.randint
    start = time.perf_counter()
    for _ in range(ops):
        r1, c1 = rand(0, rows - 1), rand(0, cols - 1)
        r2, c2 = rand(0, rows - 1), rand(0, cols - 1)
        ds.union(r1, c1, r2, c2)
    union_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(ops):
        r1, c1 = rand(0, rows - 1), rand(0, cols - 1)
        r2, c2 = rand(0, rows - 1), rand(0, cols - 1)
        ds.connected(r1, c1, r2, c2)
    find_time = time.perf_counter() - start
    return union_time, find_time


if __name__ == '__main__':
    # Run unit tests
    unittest.main(exit=False)

    # Simple demo
    demo = DisjointSetMatrix(5, 5)
    demo.union(0, 0, 0, 1)
    demo.union(0, 1, 1, 1)
    demo.union(3, 3, 4, 4)
    print("\nDemo:")
    print("Connected (0,0)-(1,1):", demo.connected(0, 0, 1, 1))
    print("Set members of (0,0):", demo.get_set(0, 0))
    print("Set members of (3,3):", demo.get_set(3, 3))

    # Benchmark (lightweight, runs quickly)
    u_time, f_time = benchmark(rows=100, cols=100, ops=20_000)
    print(f"\nBenchmark (100x100, 20k ops): union {u_time:.4f}s, find {f_time:.4f}s")
