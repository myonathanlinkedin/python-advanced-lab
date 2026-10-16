import unittest
import time
from typing import List

from core import MMatrixScaler


def _row_sums(matrix: List[List[float]]) -> List[float]:
    return [sum(row) for row in matrix]


def _col_sums(matrix: List[List[float]]) -> List[float]:
    n = len(matrix)
    return [sum(matrix[i][j] for i in range(n)) for j in range(n)]


class TestMMatrixScaler(unittest.TestCase):
    def setUp(self) -> None:
        self.tol = 1e-5  # looser tolerance for unit‑test assertions

    def test_identity_matrix(self) -> None:
        A = [[1.0, 0.0], [0.0, 1.0]]
        scaler = MMatrixScaler(tol=self.tol)
        B, r, c = scaler.scale(A)

        # Identity should remain unchanged; scaling vectors should be all ones.
        for i in range(2):
            self.assertAlmostEqual(r[i], 1.0, delta=self.tol)
            self.assertAlmostEqual(c[i], 1.0, delta=self.tol)
            for j in range(2):
                self.assertAlmostEqual(B[i][j], A[i][j], delta=self.tol)

        # Row/column sums must be 1.
        for s in _row_sums(B):
            self.assertAlmostEqual(s, 1.0, delta=self.tol)
        for s in _col_sums(B):
            self.assertAlmostEqual(s, 1.0, delta=self.tol)

    def test_simple_positive_matrix(self) -> None:
        # A symmetric positive matrix with total support.
        A = [
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ]
        scaler = MMatrixScaler(tol=self.tol, max_iter=5000)
        B, r, c = scaler.scale(A)

        # Verify scaling vectors are positive.
        for val in r + c:
            self.assertGreater(val, 0.0)

        # Verify doubly stochastic property.
        for s in _row_sums(B):
            self.assertAlmostEqual(s, 1.0, delta=self.tol)
        for s in _col_sums(B):
            self.assertAlmostEqual(s, 1.0, delta=self.tol)

        # Verify that B is close to the theoretical doubly stochastic matrix.
        # For this symmetric case, the result should be symmetric.
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(B[i][j], B[j][i], delta=self.tol)

    def test_invalid_inputs(self) -> None:
        scaler = MMatrixScaler(tol=self.tol)

        # Non‑square matrix
        with self.assertRaises(ValueError):
            scaler.scale([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Negative entry
        with self.assertRaises(ValueError):
            scaler.scale([[1.0, -0.5], [0.5, 1.0]])

        # Zero diagonal entry
        with self.assertRaises(ValueError):
            scaler.scale([[0.0, 1.0], [1.0, 2.0]])

    def test_convergence_failure(self) -> None:
        # Construct a matrix that is not scalable (zero row).
        A = [[0.0, 0.0], [1.0, 1.0]]
        scaler = MMatrixScaler(max_iter=50, tol=self.tol)
        with self.assertRaises(RuntimeError):
            scaler.scale(A)


def benchmark_scaler(size: int = 200, iterations: int = 5) -> None:
    """Simple benchmark: scale a random dense positive matrix."""
    import random

    random.seed(0)
    A = [[random.random() + 0.1 for _ in range(size)] for _ in range(size)]
    scaler = MMatrixScaler(tol=1e-4, max_iter=10_000)

    start = time.perf_counter()
    for _ in range(iterations):
        scaler.scale(A)
    elapsed = time.perf_counter() - start
    print(f"Benchmark: {size}×{size} matrix scaled {iterations} times in {elapsed:.3f}s")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run a modest benchmark to demonstrate performance (optional)
    print("\nRunning a quick benchmark (may be omitted on very small machines)...")
    benchmark_scaler(size=100, iterations=3)
