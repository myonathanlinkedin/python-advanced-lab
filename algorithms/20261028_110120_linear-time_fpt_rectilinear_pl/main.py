"""
main.py
---------

Unit‑test suite for the rectilinear planarity tester of series‑parallel graphs.
The tests cover:

* Simple base cases (single edge, triangle – not SP).
* Classic series‑parallel constructions (ladder, parallel edges).
* Degree‑constraint violations.
* Random SP graphs generated via recursive series/parallel composition.

Running the module executes the test suite and prints a short demo.
"""

import unittest
import random
from typing import Tuple, List

# Import from core.py (assumed to be in the same directory)
from core import Graph, is_rectilinear_planar, is_series_parallel


def build_series(g1: Graph, g2: Graph, s1: int, t1: int, s2: int, t2: int) -> Tuple[Graph, int, int]:
    """
    Construct the series composition of two SP graphs.
    The terminal t1 of g1 is identified with s2 of g2.
    Returns the combined graph and its new terminals (s1, t2).
    """
    offset = g1.num_vertices
    g = Graph(g1.num_vertices + g2.num_vertices - 1)  # merge the identified vertex
    # Copy edges of g1
    for u, v in g1.edges():
        g.add_edge(u, v)
    # Copy edges of g2 with offset, except the identified vertex
    for u, v in g2.edges():
        if u == s2:
            u = t1
        elif u != s2:
            u += offset
        if v == s2:
            v = t1
        elif v != s2:
            v += offset
        if u != v:
            g.add_edge(u, v)
    return g, s1, t2


def build_parallel(g1: Graph, g2: Graph, s1: int, t1: int, s2: int, t2: int) -> Tuple[Graph, int, int]:
    """
    Construct the parallel composition of two SP graphs.
    Terminals (s1, t1) and (s2, t2) are identified pairwise.
    Returns the combined graph and its terminals (s, t).
    """
    # New vertex indices: keep s1 as s, t1 as t, shift g2's internal vertices.
    offset = g1.num_vertices
    g = Graph(g1.num_vertices + g2.num_vertices - 2)  # merge both terminal pairs
    # Copy g1 edges
    for u, v in g1.edges():
        g.add_edge(u, v)
    # Copy g2 edges with offset, mapping its terminals to s1 and t1
    for u, v in g2.edges():
        if u == s2:
            u = s1
        elif u == t2:
            u = t1
        else:
            u += offset - 2
        if v == s2:
            v = s1
        elif v == t2:
            v = t1
        else:
            v += offset - 2
        if u != v:
            g.add_edge(u, v)
    return g, s1, t1


def random_sp_graph(depth: int, max_degree: int = 4) -> Tuple[Graph, int, int]:
    """
    Recursively generate a random series‑parallel graph.
    The recursion depth controls size; degree never exceeds max_degree.
    """
    if depth == 0:
        g = Graph(2)
        g.add_edge(0, 1)
        return g, 0, 1

    # Choose composition type
    if random.random() < 0.5:
        # Series
        g1, s1, t1 = random_sp_graph(depth - 1, max_degree)
        g2, s2, t2 = random_sp_graph(depth - 1, max_degree)
        g, s, t = build_series(g1, g2, s1, t1, s2, t2)
    else:
        # Parallel
        g1, s1, t1 = random_sp_graph(depth - 1, max_degree)
        g2, s2, t2 = random_sp_graph(depth - 1, max_degree)
        g, s, t = build_parallel(g1, g2, s1, t1, s2, t2)

    # Enforce degree bound by possibly removing random edges (rare)
    # This step is only to keep the generator safe; it does not affect
    # correctness of the test suite.
    for v in range(g.num_vertices):
        while g.degree(v) > max_degree:
            # Remove a random incident edge
            w = next(iter(g.adj[v]))
            g.adj[v].remove(w)
            g.adj[w].remove(v)
    return g, s, t


class TestSeriesParallel(unittest.TestCase):
    def test_single_edge(self):
        g = Graph(2)
        g.add_edge(0, 1)
        self.assertTrue(is_series_parallel(g, 0, 1))
        self.assertTrue(is_rectilinear_planar(g, 0, 1))

    def test_triangle_not_sp(self):
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        self.assertFalse(is_series_parallel(g, 0, 2))
        self.assertFalse(is_rectilinear_planar(g, 0, 2))

    def test_ladder_graph(self):
        # Ladder with 4 rungs (8 vertices)
        g = Graph(8)
        # Left side
        for i in range(0, 7, 2):
            g.add_edge(i, i + 2)
        # Right side
        for i in range(1, 8, 2):
            g.add_edge(i, i + 2)
        # Rungs
        for i in range(0, 8, 2):
            g.add_edge(i, i + 1)
        self.assertTrue(is_series_parallel(g, 0, 7))
        self.assertTrue(is_rectilinear_planar(g, 0, 7))

    def test_degree_violation(self):
        g = Graph(5)
        # Create a K_{2,3} which is series‑parallel but has degree 3 only.
        # Then add a fourth edge to a vertex to make degree 5.
        edges = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (0, 1)]
        for u, v in edges:
            g.add_edge(u, v)
        # Now vertex 0 has degree 5 > 4
        self.assertTrue(is_series_parallel(g, 0, 1))
        self.assertFalse(is_rectilinear_planar(g, 0, 1))

    def test_random_sp(self):
        for _ in range(30):
            g, s, t = random_sp_graph(depth=4)
            self.assertTrue(is_series_parallel(g, s, t))
            # Degree bound is enforced in generator
            self.assertTrue(is_rectilinear_planar(g, s, t))


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demo: construct a small series‑parallel graph and display result.
    print("\nDemo:")
    base, s, t = random_sp_graph(depth=3)
    print(f"Generated graph with {base.num_vertices} vertices, terminals ({s}, {t})")
    print(f"Series‑parallel? {is_series_parallel(base, s, t)}")
    print(f"Rectilinear planar? {is_rectilinear_planar(base, s, t)}")
# End of main.py
