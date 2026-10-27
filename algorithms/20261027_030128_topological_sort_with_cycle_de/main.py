import unittest
from typing import Dict, List

from core import CycleError, topological_sort


class TestTopologicalSort(unittest.TestCase):
    def test_empty_graph(self) -> None:
        graph: Dict[int, List[int]] = {}
        self.assertEqual(topological_sort(graph), [])

    def test_single_node(self) -> None:
        graph = {1: []}
        self.assertEqual(topological_sort(graph), [1])

    def test_simple_dag(self) -> None:
        graph = {
            5: [2, 0],
            4: [0, 1],
            2: [3],
            3: [1],
            0: [],
            1: [],
        }
        order = topological_sort(graph)
        # Verify ordering constraints
        self.assertTrue(order.index(5) < order.index(2))
        self.assertTrue(order.index(5) < order.index(0))
        self.assertTrue(order.index(4) < order.index(0))
        self.assertTrue(order.index(4) < order.index(1))
        self.assertTrue(order.index(2) < order.index(3))
        self.assertTrue(order.index(3) < order.index(1))

    def test_multiple_components(self) -> None:
        graph = {
            'a': ['b'],
            'b': [],
            'c': ['d'],
            'd': [],
            'e': [],  # isolated node
        }
        order = topological_sort(graph)
        # All nodes must appear
        self.assertCountEqual(order, ['a', 'b', 'c', 'd', 'e'])
        # Component constraints
        self.assertTrue(order.index('a') < order.index('b'))
        self.assertTrue(order.index('c') < order.index('d'))

    def test_cycle_detection(self) -> None:
        graph = {
            1: [2],
            2: [3],
            3: [1],  # cycle 1->2->3->1
            4: [5],
            5: [],
        }
        with self.assertRaises(CycleError) as cm:
            topological_sort(graph)
        self.assertSetEqual(cm.exception.cycle_nodes, {1, 2, 3})

    def test_self_loop(self) -> None:
        graph = {
            'x': ['x'],  # self‑cycle
        }
        with self.assertRaises(CycleError) as cm:
            topological_sort(graph)
        self.assertSetEqual(cm.exception.cycle_nodes, {'x'})

    def test_large_dag(self) -> None:
        # Build a DAG of 100 nodes where each node i points to i+1
        n = 100
        graph = {i: [i + 1] for i in range(n - 1)}
        graph[n - 1] = []  # last node has no outgoing edges
        order = topological_sort(graph)
        self.assertEqual(order, list(range(n)))

    def test_non_consecutive_keys(self) -> None:
        graph = {
            10: [20],
            20: [30],
            30: [],
            5: [10],
        }
        order = topological_sort(graph)
        self.assertTrue(order.index(5) < order.index(10) < order.index(20) < order.index(30))


if __name__ == '__main__':
    unittest.main()
