"""
main.py
---------

Unit‑test suite and simple demonstration for the DirectedGraph implementation.
"""

import unittest
from typing import List

from core import DirectedGraph, CycleError


class TestDirectedGraphTopologicalSort(unittest.TestCase):
    def test_empty_graph(self) -> None:
        g = DirectedGraph()
        self.assertEqual(g.topological_sort(), [])

    def test_single_node(self) -> None:
        g = DirectedGraph()
        g.add_node("A")
        self.assertEqual(g.topological_sort(), ["A"])

    def test_simple_dag(self) -> None:
        # A → B → C
        g = DirectedGraph([("A", "B"), ("B", "C")])
        order = g.topological_sort()
        self.assertEqual(order, ["A", "B", "C"])

    def test_multiple_sources(self) -> None:
        #   A   B
        #    \ /
        #     C
        g = DirectedGraph([("A", "C"), ("B", "C")])
        order = g.topological_sort()
        # A and B can appear in any order before C.
        self.assertIn(order[0], {"A", "B"})
        self.assertIn(order[1], {"A", "B"})
        self.assertNotEqual(order[0], order[1])
        self.assertEqual(order[2], "C")

    def test_disconnected_components(self) -> None:
        # Component 1: X → Y
        # Component 2: A → B → C
        g = DirectedGraph([("X", "Y"), ("A", "B"), ("B", "C")])
        order = g.topological_sort()
        # Verify relative ordering inside each component.
        self.assertTrue(order.index("X") < order.index("Y"))
        self.assertTrue(order.index("A") < order.index("B") < order.index("C"))
        # All nodes must be present.
        self.assertCountEqual(order, ["X", "Y", "A", "B", "C"])

    def test_cycle_detection_simple(self) -> None:
        # A → B → A (cycle)
        g = DirectedGraph([("A", "B"), ("B", "A")])
        with self.assertRaises(CycleError) as cm:
            g.topological_sort()
        self.assertIn("A", str(cm.exception))
        self.assertIn("B", str(cm.exception))

    def test_cycle_detection_complex(self) -> None:
        # A → B → C → D → B (cycle involving B, C, D)
        g = DirectedGraph([("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")])
        with self.assertRaises(CycleError) as cm:
            g.topological_sort()
        msg = str(cm.exception)
        self.assertTrue({"B", "C", "D"}.issubset(set(msg)))

    def test_self_loop(self) -> None:
        g = DirectedGraph([("A", "A")])
        with self.assertRaises(CycleError):
            g.topological_sort()


def benchmark_topological_sort() -> None:
    """
    Simple benchmark: generate a linear chain of N nodes and sort it.
    Prints the time taken; not part of unit tests.
    """
    import time

    N = 100_000
    edges = [(i, i + 1) for i in range(N)]
    g = DirectedGraph(edges)
    start = time.perf_counter()
    order = g.topological_sort()
    elapsed = time.perf_counter() - start
    assert len(order) == N + 1  # includes node N as sink
    print(f"Topological sort of {N+1} nodes took {elapsed:.4f} seconds.")


if __name__ == "__main__":
    # Run unit tests.
    unittest.main(exit=False)

    # Optional demo benchmark (comment out if not desired).
    # benchmark_topological_sort()
