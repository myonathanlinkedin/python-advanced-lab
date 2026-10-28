import unittest
from typing import List, Tuple

from core import HopcroftKarp


class TestHopcroftKarp(unittest.TestCase):
    def _run(self,
             left: List[int],
             right: List[int],
             edges: List[Tuple[int, int]],
             expected_size: int,
             expected_match: dict[int, int] | None = None) -> None:
        hk = HopcroftKarp(left, right, edges)
        matching = hk.maximum_matching()
        self.assertEqual(len(matching), expected_size,
                         f"Expected matching size {expected_size}, got {len(matching)}")
        if expected_match is not None:
            self.assertEqual(matching, expected_match)

    def test_empty_graph(self) -> None:
        self._run([], [], [], 0)

    def test_no_edges(self) -> None:
        self._run([1, 2], [3, 4], [], 0)

    def test_single_edge(self) -> None:
        self._run([1], [2], [(1, 2)], 1, {1: 2})

    def test_multiple_disjoint_edges(self) -> None:
        left = [1, 2, 3]
        right = [4, 5, 6]
        edges = [(1, 4), (2, 5), (3, 6)]
        self._run(left, right, edges, 3, {1: 4, 2: 5, 3: 6})

    def test_overlapping_edges(self) -> None:
        left = [1, 2]
        right = [3, 4]
        edges = [(1, 3), (1, 4), (2, 3)]
        # Maximum matching size is 2; one possible matching is {1:4, 2:3}
        hk = HopcroftKarp(left, right, edges)
        matching = hk.maximum_matching()
        self.assertEqual(len(matching), 2)
        # Verify that each left vertex is matched to a neighbour
        for u, v in matching.items():
            self.assertIn((u, v), edges)

    def test_unbalanced_partitions(self) -> None:
        left = [1, 2, 3, 4]
        right = [5, 6]
        edges = [(1, 5), (2, 5), (3, 6), (4, 6)]
        self._run(left, right, edges, 2)

    def test_complex_graph(self) -> None:
        left = [1, 2, 3, 4, 5]
        right = [6, 7, 8, 9]
        edges = [
            (1, 6), (1, 7),
            (2, 6),
            (3, 7), (3, 8),
            (4, 8), (4, 9),
            (5, 9)
        ]
        hk = HopcroftKarp(left, right, edges)
        matching = hk.maximum_matching()
        self.assertEqual(len(matching), 4)
        # Ensure matching respects adjacency
        for u, v in matching.items():
            self.assertIn((u, v), edges)

    def test_invalid_edge_raises(self) -> None:
        left = [1]
        right = [2]
        edges = [(1, 3)]  # 3 not in right partition
        with self.assertRaises(ValueError):
            HopcroftKarp(left, right, edges)

    def test_non_integer_vertices(self) -> None:
        left = ['a', 'b']
        right = ['x', 'y']
        edges = [('a', 'x'), ('b', 'y'), ('a', 'y')]
        hk = HopcroftKarp(left, right, edges)
        matching = hk.maximum_matching()
        self.assertEqual(len(matching), 2)
        for u, v in matching.items():
            self.assertIn((u, v), edges)


if __name__ == '__main__':
    unittest.main()
