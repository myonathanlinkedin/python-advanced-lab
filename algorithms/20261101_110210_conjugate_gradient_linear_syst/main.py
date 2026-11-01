"""
Unit‑test suite and demonstration for the Conjugate Gradient solver.
"""

import unittest
import math
import time
from typing import List

from core import cg_solve, Matrix, Vector


def _gaussian_elimination(A: Matrix, b: Vector) -> Vector:
    """Simple dense Gaussian elimination (no pivoting) for verification."""
    n = len(A)
    # Make copies to avoid mutating inputs
    M = [row[:] for row in A]
    y = b[:]

    # Forward elimination
    for i in range(n):
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError("Zero pivot encountered in Gaussian elimination.")
        for j in range(i + 1, n):
            factor = M[j][i] / pivot
            for k in range(i, n):
                M[j][k] -= factor * M[i][k]
            y[j] -= factor * y[i]

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / M[i][i]
    return x


class TestConjugateGradient(unittest.TestCase):
    def assertVectorAlmostEqual(self, v1: Vector, v2: Vector, places: int = 7) -> None:
        self.assertEqual(len(v1), len(v2), "Vector lengths differ.")
        for a, b in zip(v1, v2):
            self.assertAlmostEqual(a, b, places=places)

    def test_small_spd_system(self) -> None:
        # 2×2 SPD matrix with known solution
        A: Matrix = [[4.0, 1.0],
                     [1.0, 3.0]]
        x_true: Vector = [1.0, 2.0]
        b = [sum(A[i][j] * x_true[j] for j in range(2)) for i in range(2)]

        x_cg, iters = cg_solve(A, b, tol=1e-12)
        self.assertVectorAlmostEqual(x_cg, x_true, places=10)
        self.assertLessEqual(iters, 2)

    def test_random_spd_3x3(self) -> None:
        # Construct SPD matrix via Mᵀ·M + n·I
        M: Matrix = [[2.0, -1.0, 0.0],
                     [0.0,  1.5, -0.5],
                     [1.0,  0.0,  1.0]]
        n = 3
        A = [[sum(M[k][i] * M[k][j] for k in range(n)) + (n if i == j else 0)
              for j in range(n)] for i in range(n)]

        x_true: Vector = [0.5, -1.2, 3.3]
        b = [sum(A[i][j] * x_true[j] for j in range(n)) for i in range(n)]

        x_cg, iters = cg_solve(A, b, tol=1e-10)
        self.assertVectorAlmostEqual(x_cg, x_true, places=8)
        self.assertLessEqual(iters, n)

    def test_zero_rhs(self) -> None:
        A: Matrix = [[2.0, 0.0],
                     [0.0, 3.0]]
        b: Vector = [0.0, 0.0]
        x_cg, iters = cg_solve(A, b)
        self.assertVectorAlmostEqual(x_cg, [0.0, 0.0])
        self.assertEqual(iters, 0)

    def test_invalid_dimensions(self) -> None:
        A: Matrix = [[1.0, 2.0], [3.0, 4.0]]
        b: Vector = [1.0, 2.0, 3.0]  # mismatched length
        with self.assertRaises(ValueError):
            cg_solve(A, b)

    def test_non_spd_matrix_raises(self) -> None:
        # Indefinite matrix (eigenvalues of opposite sign)
        A: Matrix = [[0.0, 1.0],
                     [1.0, 0.0]]
        b: Vector = [1.0, 1.0]
        with self.assertRaises(ValueError):
            cg_solve(A, b, max_iter=5)

    def test_against_gaussian_elimination(self) -> None:
        # 4×4 random SPD matrix
        import random
        random.seed(0)

        n = 4
        M = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
        A = [[sum(M[k][i] * M[k][j] for k in range(n)) + n * (i == j)
              for j in range(n)] for i in range(n)]

        x_true = [random.uniform(-5, 5) for _ in range(n)]
        b = [sum(A[i][j] * x_true[j] for j in range(n)) for i in range(n)]

        x_cg, _ = cg_solve(A, b, tol=1e-12)
        x_ge = _gaussian_elimination(A, b)

        self.assertVectorAlmostEqual(x_cg, x_ge, places=7)

    def test_performance_demo(self) -> None:
        # Simple timing for a 100×100 SPD system (still deterministic)
        n = 100
        # Build SPD matrix: A = I + L·Lᵀ where L is lower‑triangular with 0.1 entries
        L = [[0.1 if i > j else 0.0 for j in range(n)] for i in range(n)]
        A = [[sum(L[k][i] * L[k][j] for k in range(n)) + (1.0 if i == j else 0.0)
              for j in range(n)] for i in range(n)]

        x_true = [1.0] * n
        b = [sum(A[i][j] * x_true[j] for j in range(n)) for i in range(n)]

        start = time.perf_counter()
        x_cg, iters = cg_solve(A, b, tol=1e-8)
        duration = time.perf_counter() - start

        # Verify residual is within tolerance
        residual_norm = math.sqrt(sum((b[i] - sum(A[i][j] * x_cg[j] for j in range(n))) ** 2 for i in range(n)))
        self.assertLessEqual(residual_norm / math.sqrt(sum(bi * bi for bi in b)), 1e-8)

        # Simple sanity check on iteration count (should be <= n)
        self.assertLessEqual(iters, n)

        # Print a short performance summary (allowed in test output)
        print(f"\nCG solved 100×100 SPD system in {duration:.4f}s with {iters} iterations.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
