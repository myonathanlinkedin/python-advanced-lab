"""
Unit‑test suite for the Conjugate Gradient implementation in `core.py`.

The tests cover:
- Correctness on a small analytically solvable system.
- Accuracy on randomly generated SPD matrices.
- Proper handling of invalid inputs.
"""

import unittest
import random
import math
from typing import List

from core import conjugate_gradient, is_spd, Matrix, Vector


def _gaussian_elimination(A: Matrix, b: Vector) -> Vector:
    """
    Simple dense Gaussian elimination with partial pivoting.
    Returns the exact solution for small systems (used only in tests).
    """
    n = len(A)
    # Build augmented matrix
    M = [row[:] + [b_i] for row, b_i in zip(A, b)]

    # Forward elimination
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


def _generate_spd_matrix(n: int, seed: int = 0) -> Matrix:
    """Generate a random n×n symmetric positive‑definite matrix."""
    random.seed(seed)
    # Random dense matrix
    M = [[random.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]
    # Form A = Mᵀ·M + n·I to guarantee SPD
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = sum(M[k][i] * M[k][j] for k in range(n))
            if i == j:
                val += n  # shift diagonal
            A[i][j] = val
            A[j][i] = val
    return A


class TestConjugateGradient(unittest.TestCase):
    def test_small_known_system(self):
        """Solve a 2×2 SPD system with a known analytical solution."""
        A = [[4.0, 1.0],
             [1.0, 3.0]]
        b = [1.0, 2.0]
        expected = _gaussian_elimination(A, b)

        x, iters, resid = conjugate_gradient(A, b, tol=1e-12)
        for xi, ei in zip(x, expected):
            self.assertAlmostEqual(xi, ei, places=10)
        self.assertLess(resid, 1e-12)

    def test_random_spd_accuracy(self):
        """Check that CG attains a small residual on random SPD matrices."""
        for n in (3, 5, 8):
            A = _generate_spd_matrix(n, seed=n)
            self.assertTrue(is_spd(A))
            # Random right‑hand side
            b = [random.uniform(-5.0, 5.0) for _ in range(n)]
            x, iters, resid = conjugate_gradient(A, b, tol=1e-10, max_iter=5 * n)
            # Verify residual norm
            r = [bi - ai for bi, ai in zip(b, [sum(Ai[j] * xj for j, xj in enumerate(x)) for Ai in A])]
            self.assertLess(norm(r), 1e-8)

    def test_invalid_non_spd(self):
        """A non‑SPD matrix must raise a ValueError."""
        A = [[0.0, 1.0],
             [1.0, 0.0]]  # indefinite
        b = [1.0, 1.0]
        with self.assertRaises(ValueError):
            conjugate_gradient(A, b)

    def test_dimension_mismatch(self):
        """Mismatched dimensions between A and b should raise an error."""
        A = [[2.0, 0.0], [0.0, 2.0]]
        b = [1.0]  # wrong size
        with self.assertRaises(ValueError):
            conjugate_gradient(A, b)

    def test_zero_initial_guess_convergence(self):
        """Ensure that providing an explicit zero initial guess behaves like default."""
        A = [[2.0, 0.0], [0.0, 2.0]]
        b = [4.0, 6.0]
        x0 = [0.0, 0.0]
        x1, it1, _ = conjugate_gradient(A, b, x0=None)
        x2, it2, _ = conjugate_gradient(A, b, x0=x0)
        self.assertEqual(it1, it2)
        for a, b_ in zip(x1, x2):
            self.assertAlmostEqual(a, b_, places=12)


def norm(v: List[float]) -> float:
    """Utility wrapper for Euclidean norm used in tests."""
    return math.sqrt(sum(x * x for x in v))


if __name__ == '__main__':
    unittest.main()
