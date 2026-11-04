import unittest
import random
import time
from typing import Dict, Set, Any

import core


class TestLabelConnected(unittest.TestCase):
    def test_empty_graph(self):
        adj: Dict[Any, Set[Any]] = {}
        labels: Dict[Any, Any] = {}
        self.assertTrue(core.is_label_connected(adj, labels))

    def test_single_vertex(self):
        adj = {0: set()}
        labels = {0: "a"}
        self.assertTrue(core.is_label_connected(adj, labels))

    def test_connected_all_same_label(self):
        adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        labels = {0: "a", 1: "a", 2: "a"}
        self.assertTrue(core.is_label_connected(adj, labels))

    def test_connected_multiple_labels_connected(self):
        adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        labels = {0: "a", 1: "a", 2: "b", 3: "b"}
        self.assertTrue(core.is_label_connected(adj, labels))

    def test_connected_multiple_labels_not_connected(self):
        adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        labels = {0: "a", 1: "b", 2: "a", 3: "b"}
        self.assertFalse(core.is_label_connected(adj, labels))

    def test_disconnected_graph_connected_labels(self):
        adj = {0: {1}, 1: {0}, 2: {3}, 3: {2}}
        labels = {0: "a", 1: "a", 2: "b", 3: "b"}
        self.assertTrue(core.is_label_connected(adj, labels))

    def test_disconnected_graph_not_connected_label(self):
        adj = {0: {1}, 1: {0}, 2: {3}, 3: {2}}
        labels = {0: "a", 1: "b", 2: "a", 3: "b"}
        self.assertFalse(core.is_label_connected(adj, labels))

    def test_vertices_mismatch_raises(self):
        adj = {0: {1}, 1: {0}}
        labels = {0: "a"}  # missing vertex 1
        with self.assertRaises(ValueError):
            core.is_label_connected(adj, labels)

    def test_large_random_graph(self):
        n = 200
        m = 500
        labels_list = [f"label_{i}" for i in range(5)]
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
        labels: Dict[int, str] = {}
        for i in range(n):
            labels[i] = random.choice(labels_list)

        # Build random edges
        edges = set()
        while len(edges) < m:
            u, v = random.sample(range(n), 2)
            if u > v:
                u, v = v, u
            edges.add((u, v))
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # The function should run without error
        result = core.is_label_connected(adj, labels)
        self.assertIsInstance(result, bool)

    def test_performance_small(self):
        # Simple performance check for small graph
        n = 1000
        adj = {i: set() for i in range(n)}
        labels = {i: "a" for i in range(n)}
        for i in range(n - 1):
            adj[i].add(i + 1)
            adj[i + 1].add(i)
        start = time.perf_counter()
        self.assertTrue(core.is_label_connected(adj, labels))
        duration = time.perf_counter() - start
        self.assertLess(duration, 0.1, f"Execution took too long: {duration}s")


def demo():
    """
    Simple demo showing usage of is_label_connected.
    """
    adj = {
        0: {1, 2},
        1: {0, 3},
        2: {0},
        3: {1},
        4: {5},
        5: {4},
    }
    labels = {
        0: "red",
        1: "red",
        2: "red",
        3: "blue",
        4: "blue",
        5: "blue",
    }
    print("Graph adjacency:", adj)
    print("Labels:", labels)
    print("Is label-connected?", core.is_label_connected(adj, labels))


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    print("\nDemo:")
    demo()
