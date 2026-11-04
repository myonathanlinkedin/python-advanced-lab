"""
main.py
---------

Unit‑test suite for the APSP and round‑trip utilities defined in ``core.py``.
The tests cover typical scenarios, edge cases, and the detection of negative
cycles.  Running this file executes the test harness.
"""

import unittest
from typing import List, Tuple

from core import INF, floyd_warshall, roundtrip_matrix


class TestFloydWarshall(unittest.TestCase):
    def assertMatrixEqual(
        self, got: List[List[float]], expected: List[List[float]], places: int = 7
    ) -> None:
        self.assertEqual(len(got), len(expected), "Row count mismatch")
        for i, (row_got, row_exp) in enumerate(zip(got, expected)):
            self.assertEqual(
                len(row_got), len(row_exp), f"Column count mismatch in row {i}"
            )
            for j, (val_got, val_exp) in enumerate(zip(row_got, row_exp)):
                if val_got == INF and val_exp == INF:
                    continue
                self.assertAlmostEqual(
                    val_got, val_exp, places=places, msg=f"Mismatch at ({i},{j})"
                )

    def test_simple_graph(self) -> None:
        # Graph:
        # 0 → 1 (2), 0 → 2 (5)
        # 1 → 2 (1)
        n = 3
        edges: List[Tuple[int, int, float]] = [(0, 1, 2.0), (0, 2, 5.0), (1, 2, 1.0)]
        dist = floyd_warshall(n, edges)
        expected = [
            [0.0, 2.0, 3.0],
            [INF, 0.0, 1.0],
            [INF, INF, 0.0],
        ]
        self.assertMatrixEqual(dist, expected)

    def test_unreachable_vertices(self) -> None:
        n = 4
        edges = [(0, 1, 3.0), (2, 3, 4.0)]
        dist = floyd_warshall(n, edges)
        expected = [
            [0.0, 3.0, INF, INF],
            [INF, 0.0, INF, INF],
            [INF, INF, 0.0, 4.0],
            [INF, INF, INF, 0.0],
        ]
        self.assertMatrixEqual(dist, expected)

    def test_negative_weights_no_cycle(self) -> None:
        # 0 → 1 (-2), 1 → 2 (3), 0 → 2 (4)
        n = 3
        edges = [(0, 1, -2.0), (1, 2, 3.0), (0, 2, 4.0)]
        dist = floyd_warshall(n, edges)
        expected = [
            [0.0, -2.0, 1.0],
            [INF, 0.0, 3.0],
            [INF, INF, 0.0],
        ]
        self.assertMatrixEqual(dist, expected)

    def test_negative_cycle_detection(self) -> None:
        # 0 → 1 (1), 1 → 2 (-2), 2 → 0 (-2) creates a negative cycle of total -3
        n = 3
        edges = [(0, 1, 1.0), (1, 2, -2.0), (2, 0, -2.0)]
        with self.assertRaises(ValueError):
            floyd_warshall(n, edges)

    def test_parallel_edges_keep_minimum(self) -> None:
        n = 2
        edges = [(0, 1, 10.0), (0, 1, 5.0), (0, 1, 7.0)]
        dist = floyd_warshall(n, edges)
        expected = [
            [0.0, 5.0],
            [INF, 0.0],
        ]
        self.assertMatrixEqual(dist, expected)


class TestRoundtripMatrix(unittest.TestCase):
    def test_basic_roundtrip(self) -> None:
        n = 3
        edges = [(0, 1, 2.0), (1, 0, 3.0), (1, 2, 1.0), (2, 1, 4.0)]
        dist = floyd_warshall(n, edges)
        rt = roundtrip_matrix(dist)
        expected = [
            [0.0, 5.0, INF],
            [5.0, 0.0, 5.0],
            [INF, 5.0, 0.0],
        ]
        # Use same helper as before
        self.assertEqual(len(rt), len(expected))
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(rt[i][j], expected[i][j], places=7)

    def test_unreachable_roundtrip(self) -> None:
        n = 2
        edges = [(0, 1, 1.0)]  # No edge back from 1 to 0
        dist = floyd_warshall(n, edges)
        rt = roundtrip_matrix(dist)
        expected = [
            [0.0, INF],
            [INF, 0.0],
        ]
        self.assertEqual(rt, expected)


if __name__ == "__main__":
    unittest.main()
