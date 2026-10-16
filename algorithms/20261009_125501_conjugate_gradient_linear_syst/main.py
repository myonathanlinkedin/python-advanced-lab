"""
Unit‑test suite for the Conjugate Gradient implementation.

Covers:
- Small dense SPD matrices (2×2, 3×3).
- Callable matrix‑vector product interface.
- Edge cases: zero RHS, singular detection, tolerance enforcement.
"""

import unittest
import math
from typing import List

# Import the solver from core.py
from core import ConjugateGradient, _dot, _norm


def _gaussian_elimination(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Simple (non‑optimized) Gaussian elimination with partial pivoting.
    Used only for verification in unit tests.
    """
    n = len(A)
    # Deep copy to avoid mutating inputs
    M = [row[:] for row in A]
    rhs = b[:]

    for k in range(n):
        # Pivot
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[max_row][k]) < 1e-12:
            raise ValueError("Matrix is singular to working precision.")
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
            rhs[k], rhs[max_row] = rhs[max_row], rhs[k]

        # Eliminate
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n):
                M[i][j] -= factor * M[k][j]
            rhs[i] -= factor * rhs[k]

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (rhs[i] - s) / M[i][i]
    return x


class TestConjugateGradient(unittest.TestCase):
    def assertVectorAlmostEqual(self, v1: List[float], v2: List[float], places: int = 7) -> None:
        self.assertEqual(len(v1), len(v2), "Vector lengths differ.")
        for a, b in zip(v1, v2):
            self.assertAlmostEqual(a, b, places=places)

    def test_2x2_spd(self):
        A = [[4.0, 1.0],
             [1.0, 3.0]]
        b = [1.0, 2.0]
        cg = ConjugateGradient(A, tol=1e-10)
        x = cg.solve(b)
        x_ref = _gaussian_elimination(A, b)
        self.assertVectorAlmostEqual(x, x_ref)

    def test_3x3_spd(self):
        A = [[6.0, 2.0, 1.0],
             [2.0, 5.0, 2.0],
             [1.0, 2.0, 4.0]]
        b = [7.0, -8.0, 6.0]
        cg = ConjugateGradient(A, tol=1e-12)
        x = cg.solve(b)
        x_ref = _gaussian_elimination(A, b)
        self.assertVectorAlmostEqual(x, x_ref)

    def test_callable_interface(self):
        # Same matrix as test_2x2_spd but supplied as a function.
        A_dense = [[4.0, 1.0],
                   [1.0, 3.0]]

        def matvec(v: List[float]) -> List[float]:
            return [_dot(row, v) for row in A_dense]

        b = [1.0, 2.0]
        cg = ConjugateGradient(matvec, tol=1e-10)
        x = cg.solve(b)
        x_ref = _gaussian_elimination(A_dense, b)
        self.assertVectorAlmostEqual(x, x_ref)

    def test_zero_rhs(self):
        A = [[2.0, 0.5],
             [0.5, 1.0]]
        b = [0.0, 0.0]
        cg = ConjugateGradient(A)
        x = cg.solve(b, x0=[1.0, -1.0])  # non‑zero initial guess
        self.assertVectorAlmostEqual(x, [0.0, 0.0])

    def test_non_spd_detection(self):
        # Matrix with a negative eigenvalue (not SPD)
        A = [[0.0, 1.0],
             [1.0, 0.0]]
        b = [1.0, 1.0]
        cg = ConjugateGradient(A, tol=1e-6, max_iter=10)
        with self.assertRaises(ValueError):
            cg.solve(b)

    def test_tolerance_respects_relative_residual(self):
        A = [[10.0, 2.0],
             [2.0, 5.0]]
        b = [1.0, 1.0]
        # Very loose tolerance should stop early.
        cg = ConjugateGradient(A, tol=1e-1, max_iter=1000)
        x = cg.solve(b)
        # Verify that the residual norm is indeed within the requested tolerance.
        residual = [bi - ai for bi, ai in zip(b, [_dot(row, x) for row in A])]
        self.assertLessEqual(_norm(residual), 1e-1 * _norm(b))

    def test_max_iterations_limit(self):
        A = [[4.0, 1.0],
             [1.0, 3.0]]
        b = [1.0, 2.0]
        cg = ConjugateGradient(A, tol=1e-12, max_iter=1)  # force early stop
        x = cg.solve(b)
        # With only one iteration, solution cannot be exact; check that it differs.
        x_ref = _gaussian_elimination(A, b)
        diff = _norm([xi - ri for xi, ri in zip(x, x_ref)])
        self.assertGreater(diff, 1e-3)


if __name__ == "__main__":
    # Run the unit tests with verbose output.
    unittest.main(verbosity=2)
