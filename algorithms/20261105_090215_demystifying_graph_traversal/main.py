"""
main.py
---------

Unit‑test suite and simple benchmark for the graph traversal implementation
found in core.py.  The tests cover typical edge cases and verify deterministic
ordering for both BFS and DFS on small graphs.
"""

import unittest
import timeit
from typing import List

from core import Graph, bfs, dfs, dfs_iter


class TestGraphTraversals(unittest.TestCase):
    def setUp(self) -> None:
        # Simple undirected graph used by several tests
        self.g = Graph[int]()
        edges = [
            (1, 2), (1, 3), (2, 4), (2, 5),
            (3, 6), (3, 7)
        ]
        for u, v in edges:
            self.g.add_edge(u, v)

    def test_empty_graph(self) -> None:
        empty = Graph[int]()
        # Adding no edges yields an empty node set
        self.assertEqual(empty.nodes(), set())
        # Traversal on an empty graph should raise ValueError
        with self.assertRaises(ValueError):
            empty.bfs(0)

    def test_single_node(self) -> None:
        single = Graph[int]()
        single.add_node(42)
        self.assertEqual(single.bfs(42), [42])
        self.assertEqual(single.dfs(42), [42])
        self.assertEqual(single.dfs_iter(42), [42])

    def test_invalid_start(self) -> None:
        with self.assertRaises(ValueError):
            self.g.bfs(999)  # node not present

    def test_bfs_order(self) -> None:
        # Expected BFS order from node 1: level‑order traversal
        expected = [1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(self.g.bfs(1), expected)

    def test_dfs_order_recursive(self) -> None:
        # Deterministic order depends on adjacency iteration order.
        # Our implementation iterates in the natural set order, which is
        # nondeterministic; however, for the constructed graph the order is
        # predictable because we inserted edges in a consistent way.
        order = self.g.dfs(1)
        # Verify that all nodes are visited and start node is first.
        self.assertEqual(order[0], 1)
        self.assertCountEqual(order, list(range(1, 8)))

    def test_dfs_order_iterative(self) -> None:
        # Iterative version pushes neighbors in reverse sorted order,
        # yielding a deterministic pre‑order traversal.
        expected = [1, 2, 4, 5, 3, 6, 7]
        self.assertEqual(self.g.dfs_iter(1), expected)

    def test_directed_cycle(self) -> None:
        dg = Graph[int](directed=True)
        dg.add_edge(1, 2)
        dg.add_edge(2, 3)
        dg.add_edge(3, 1)  # cycle
        self.assertEqual(dg.bfs(1), [1, 2, 3])
        self.assertEqual(dg.dfs_iter(1), [1, 2, 3])

    def test_path_exists(self) -> None:
        self.assertTrue(self.g.path_exists(1, 7, method="bfs"))
        self.assertTrue(self.g.path_exists(1, 7, method="dfs"))
        self.assertFalse(self.g.path_exists(4, 6, method="bfs"))  # disconnected in directed sense

    def test_functional_wrappers(self) -> None:
        self.assertEqual(bfs(self.g, 1), self.g.bfs(1))
        self.assertEqual(dfs(self.g, 1), self.g.dfs(1))
        self.assertEqual(dfs_iter(self.g, 1), self.g.dfs_iter(1))


def benchmark_traversal() -> None:
    """
    Simple micro‑benchmark comparing BFS and DFS on a larger random graph.
    The function prints timings; it is not part of the unit‑test suite.
    """
    import random

    N = 10_000
    E = 30_000
    rand = random.Random(0)

    g = Graph[int]()
    for i in range(N):
        g.add_node(i)
    for _ in range(E):
        u = rand.randrange(N)
        v = rand.randrange(N)
        g.add_edge(u, v)

    start = 0
    bfs_time = timeit.timeit(lambda: g.bfs(start), number=1)
    dfs_time = timeit.timeit(lambda: g.dfs_iter(start), number=1)

    print(f"BFS visited {len(g.bfs(start))} nodes in {bfs_time:.4f}s")
    print(f"DFS visited {len(g.dfs_iter(start))} nodes in {dfs_time:.4f}s")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run a quick benchmark (optional, not required for correctness)
    print("\n--- Simple Benchmark ---")
    benchmark_traversal()
