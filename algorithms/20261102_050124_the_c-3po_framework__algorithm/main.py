"""
main.py
---------

Unit‑test suite for the C‑3PO reference implementation.
All tests are deterministic and rely solely on the standard library.
"""

import unittest
import random
from typing import Tuple

from core import BipartiteGraph


def _random_bipartite(n_left: int, n_right: int, edge_prob: float) -> BipartiteGraph:
    """Utility to generate a random bipartite graph."""
    g = BipartiteGraph(n_left, n_right)
    for i in range(n_left):
        for j in range(n_right):
            if random.random() < edge_prob:
                g.add_edge(i, j)
    return g


class TestGF2Rank(unittest.TestCase):
    def test_zero_matrix(self):
        g = BipartiteGraph(3, 4)  # no edges
        self.assertEqual(g.left_rank_gf2(), 0)

    def test_identity_matrix(self):
        # Build a graph whose left adjacency matrix is the 3×3 identity.
        g = BipartiteGraph(3, 3)
        for i in range(3):
            g.add_edge(i, i)
        self.assertEqual(g.left_rank_gf2(), 3)

    def test_known_rank(self):
        # Matrix:
        # 1 1 0
        # 0 1 1
        # 1 0 1   -> rank 3 over GF(2)
        g = BipartiteGraph(3, 3)
        g.add_edge(0, 0)
        g.add_edge(0, 1)
        g.add_edge(1, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        g.add_edge(2, 2)
        self.assertEqual(g.left_rank_gf2(), 3)

    def test_dependent_rows(self):
        # Rows: 110, 101, 011  (third row = xor of first two)
        g = BipartiteGraph(3, 3)
        g.add_edge(0, 0)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        g.add_edge(1, 2)
        g.add_edge(2, 1)
        g.add_edge(2, 2)
        self.assertEqual(g.left_rank_gf2(), 2)


class TestIndependentSetCounting(unittest.TestCase):
    def test_empty_graph(self):
        g = BipartiteGraph(0, 0)
        self.assertEqual(g.count_independent_sets_bruteforce(), 1)  # only the empty set

    def test_isolated_vertices(self):
        g = BipartiteGraph(2, 3)  # no edges
        # Every vertex can be chosen independently: 2^2 * 2^3 = 32
        self.assertEqual(g.count_independent_sets_bruteforce(), 32)

    def test_single_edge(self):
        g = BipartiteGraph(1, 1)
        g.add_edge(0, 0)
        # Independent sets: {}, {L0}, {R0}
        self.assertEqual(g.count_independent_sets_bruteforce(), 3)

    def test_random_small_graphs(self):
        random.seed(0)
        for n_left, n_right in [(2, 2), (3, 4), (4, 3)]:
            for _ in range(20):
                g = _random_bipartite(n_left, n_right, edge_prob=0.4)
                brute = g.count_independent_sets_bruteforce()
                low = g.count_independent_sets_low_rank(rank_threshold=5)
                self.assertEqual(brute, low,
                                 msg=f"Mismatch on graph ({n_left},{n_right})")

    def test_low_rank_vs_bruteforce(self):
        # Construct a low‑rank graph: all left vertices share the same neighbour set.
        g = BipartiteGraph(5, 3)
        for i in range(5):
            g.add_edge(i, 0)
            g.add_edge(i, 2)   # neighbours {0,2} for every left vertex
        # Rank should be 1 (all rows identical)
        self.assertEqual(g.left_rank_gf2(), 1)
        brute = g.count_independent_sets_bruteforce()
        low = g.count_independent_sets_low_rank(rank_threshold=2)
        self.assertEqual(brute, low)


if __name__ == "__main__":
    # Run the unit tests with a higher verbosity for demonstration.
    unittest.main(verbosity=2)

# End of main.py
