import unittest
import random
import time
from typing import List
from core import ConjugateGradientSolver, _norm, _matvec, Matrix, Vector


def _generate_spd_matrix(n: int, seed: int = 0) -> Matrix:
    """Generate a random symmetric positive‑definite matrix of size n."""
    random.seed(seed)
    M: List[List[float]] = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    # Compute A = Mᵀ·M + n·I to guarantee SPD.
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = sum(M[k][i] * M[k][j] for k in range(n))
        A[i][i] += n  # shift eigenvalues away from zero
    return A


class TestConjugateGradient(unittest.TestCase):
    def test_small_spd(self):
        A = [[4.0, 1.0],
             [1.0, 3.0]]
        b = [1.0, 2.0]
        expected = [1.0 / 11.0, 7.0 / 11.0]  # analytical solution
        x = ConjugateGradientSolver.solve(A, b, tol=1e-12)
        for xi, ei in zip(x, expected):
            self.assertAlmostEqual(xi, ei, places=9)

    def test_zero_rhs(self):
        A = [[2.0, -1.0],
             [-1.0, 2.0]]
        b = [0.0, 0.0]
        x = ConjugateGradientSolver.solve(A, b)
        self.assertTrue(all(abs(xi) < 1e-12 for xi in x))

    def test_random_spd(self):
        n = 5
        A = _generate_spd_matrix(n, seed=42)
        x_true = [random.uniform(-5, 5) for _ in range(n)]
        b = _matvec(A, x_true)
        x_computed = ConjugateGradientSolver.solve(A, b, tol=1e-10)
        residual = _norm(_sub(_matvec(A, x_computed), b))
        self.assertLess(residual, 1e-8)

    def test_non_square_error(self):
        A = [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]]
        b = [1.0, 2.0]
        with self.assertRaises(ValueError):
            ConjugateGradientSolver.solve(A, b)

    def test_max_iter_limit(self):
        n = 10
        A = _generate_spd_matrix(n, seed=7)
        b = [random.random() for _ in range(n)]
        # Force early termination by setting max_iter very low.
        x = ConjugateGradientSolver.solve(A, b, max_iter=2, tol=1e-15)
        # Verify that we still return a vector of correct size.
        self.assertEqual(len(x), n)

    def test_convergence_speed(self):
        n = 100
        A = _generate_spd_matrix(n, seed=99)
        b = [random.random() for _ in range(n)]
        start = time.time()
        x = ConjugateGradientSolver.solve(A, b, tol=1e-8)
        duration = time.time() - start
        residual = _norm(_sub(_matvec(A, x), b))
        self.assertLess(residual, 1e-6)
        # Simple sanity check: solving 100×100 SPD should finish within a second on modest hardware.
        self.assertLess(duration, 2.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
