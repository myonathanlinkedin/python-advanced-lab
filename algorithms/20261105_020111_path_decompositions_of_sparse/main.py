"""
Unit‑test suite for the pathwidth implementation.

The tests cover:
* trivial graphs (empty, single vertex)
* classic families (paths, cycles, stars, cliques)
* random small graphs where DP result is cross‑checked against brute force
"""

import unittest
import random
from core import Graph, pathwidth, _brute_force_pathwidth
from typing import List, Tuple


def make_path(n: int) -> Graph:
    edges = [(i, i + 1) for i in range(n - 1)]
    return Graph.from_edges(n, edges)


def make_cycle(n: int) -> Graph:
    if n == 0:
        return Graph(0)
    edges = [(i, (i + 1) % n) for i in range(n)]
    return Graph.from_edges(n, edges)


def make_star(n: int) -> Graph:
    # vertex 0 is centre
    edges = [(0, i) for i in range(1, n)]
    return Graph.from_edges(n, edges)


def make_clique(n: int) -> Graph:
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return Graph.from_edges(n, edges)


class TestPathwidth(unittest.TestCase):
    def test_empty(self):
        g = Graph(0)
        self.assertEqual(pathwidth(g), 0)

    def test_single_vertex(self):
        g = Graph(1)
        self.assertEqual(pathwidth(g), 0)

    def test_path(self):
        for n in range(2, 7):
            g = make_path(n)
            self.assertEqual(pathwidth(g), 1, f"path of length {n}")

    def test_cycle(self):
        for n in range(3, 7):
            g = make_cycle(n)
            # Cycle has pathwidth 2 for n ≥ 3
            self.assertEqual(pathwidth(g), 2, f"cycle of size {n}")

    def test_star(self):
        for n in range(2, 7):
            g = make_star(n)
            self.assertEqual(pathwidth(g), 1, f"star with {n} vertices")

    def test_clique(self):
        for n in range(1, 6):
            g = make_clique(n)
            self.assertEqual(pathwidth(g), n - 1, f"clique K{n}")

    def test_random_small(self):
        # Exhaustive cross‑check for n ≤ 7
        for n in range(1, 8):
            for _ in range(30):
                edges = self._random_edges(n, p=0.4)
                g = Graph.from_edges(n, edges)
                dp = pathwidth(g)
                bf = _brute_force_pathwidth(g)
                self.assertEqual(dp, bf,
                                 f"Mismatch on random graph n={n}, edges={edges}")

    @staticmethod
    def _random_edges(n: int, p: float) -> List[Tuple[int, int]]:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    edges.append((i, j))
        return edges


if __name__ == "__main__":
    unittest.main()
