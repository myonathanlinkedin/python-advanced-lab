"""
Unit‑test suite and simple benchmark for the Conjugate Gradient implementation.
"""

import random
import unittest
from typing import List

from core import conjugate_gradient, dot, matvec

def _gaussian_elimination(A: List[List[float]], b: List[float]) -> List[float]:
    """Very small dense solver using naive Gaussian elimination with partial pivoting."""
    n = len(A)
    # Build augmented matrix
    M = [row[:] + [b_i] for row, b_i in zip(A, b)]
    for k in range(n):
        # Pivot
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[max_row][k]) < 1e-12:
            raise ValueError("Matrix is singular.")
        M[k], M[max_row] = M[max_row], M[k]
        # Eliminate
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]
    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x

def _random_spd_matrix(n: int, seed: int = 0) -> List[List[float]]:
    """Generate a random symmetric positive‑definite matrix of size n."""
    random.seed(seed)
    # Create a random dense matrix and multiply it by its transpose
    R = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = sum(R[i][k] * R[j][k] for k in range(n))
            A[i][j] = A[j][i] = val + (n if i == j else 0.0)  # ensure diagonal dominance
    return A

class TestConjugateGradient(unittest.TestCase):
    def test_small_system(self):
        A = [[4.0, 1.0],
             [1.0, 3.0]]
        b = [1.0, 2.0]
        x_expected = _gaussian_elimination(A, b)
        x_cg, iters = conjugate_gradient(A, b, tol=1e-12)
        for xc, xe in zip(x_cg, x_expected):
            self.assertAlmostEqual(xc, xe, places=9)
        self.assertLessEqual(iters, len(b))

    def test_random_spd(self):
        n = 5
        A = _random_spd_matrix(n, seed=42)
        b = [random.uniform(-5, 5) for _ in range(n)]
        x_expected = _gaussian_elimination(A, b)
        x_cg, _ = conjugate_gradient(A, b, tol=1e-10)
        for xc, xe in zip(x_cg, x_expected):
            self.assertAlmostEqual(xc, xe, places=7)

    def test_callable_matvec(self):
        n = 4
        A = _random_spd_matrix(n, seed=7)
        def mv(v: List[float]) -> List[float]:
            return matvec(A, v)
        b = [random.uniform(-2, 2) for _ in range(n)]
        x_cg, _ = conjugate_gradient(mv, b, tol=1e-10)
        x_expected = _gaussian_elimination(A, b)
        for xc, xe in zip(x_cg, x_expected):
            self.assertAlmostEqual(xc, xe, places=7)

    def test_zero_rhs(self):
        A = [[2.0, 0.0], [0.0, 3.0]]
        b = [0.0, 0.0]
        x_cg, iters = conjugate_gradient(A, b)
        self.assertEqual(x_cg, [0.0, 0.0])
        self.assertEqual(iters, 0)

    def test_invalid_dimensions(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        b = [1.0]
        with self.assertRaises(ValueError):
            conjugate_gradient(A, b)

    def test_non_spd_detection(self):
        # Indefinite matrix (eigenvalues of opposite sign)
        A = [[0.0, 1.0], [1.0, 0.0]]
        b = [1.0, 1.0]
        with self.assertRaises(ValueError):
            conjugate_gradient(A, b, max_iter=5)

def benchmark_cg(n: int = 200) -> None:
    """Simple benchmark printing iteration count and residual norm."""
    A = _random_spd_matrix(n, seed=123)
    b = [random.uniform(-1, 1) for _ in range(n)]
    x, iters = conjugate_gradient(A, b, tol=1e-8)
    # Compute residual norm
    r = [bi - ai for bi, ai in zip(b, matvec(A, x))]
    res_norm = dot(r, r) ** 0.5
    print(f"CG benchmark (n={n}): {iters} iterations, residual norm = {res_norm:e}")

if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run a modest benchmark
    print("\nRunning benchmark...")
    benchmark_cg(150)
