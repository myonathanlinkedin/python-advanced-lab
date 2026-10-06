"""
Unit tests and simple benchmark for the Conjugate Gradient implementation.
"""

import random
import time
import unittest
import math
from typing import List

from core import conjugate_gradient, Vector, Matrix


def _random_matrix(n: int, seed: int = 0) -> Matrix:
    """Generate a random dense matrix with entries in [-1, 1]."""
    rnd = random.Random(seed)
    return [[rnd.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]


def _make_spd(n: int, seed: int = 0) -> Matrix:
    """
    Construct a symmetric positive‑definite matrix A = Mᵀ·M + n·I.
    The added diagonal term guarantees strict positivity.
    """
    M = _random_matrix(n, seed)
    # Compute Mᵀ·M
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = sum(M[k][i] * M[k][j] for k in range(n))
            A[i][j] = val
            A[j][i] = val
    # Add n*I
    for i in range(n):
        A[i][i] += n
    return A


def _random_vector(n: int, seed: int = 0) -> Vector:
    rnd = random.Random(seed)
    return [rnd.uniform(-1.0, 1.0) for _ in range(n)]


def _residual_norm(A: Matrix, x: Vector, b: Vector) -> float:
    """Return the Euclidean norm of the residual r = b - A·x."""
    Ax = [sum(aij * xj for aij, xj in zip(row, x)) for row in A]
    r = [bi - axi for bi, axi in zip(b, Ax)]
    return math.sqrt(sum(ri * ri for ri in r))


class TestConjugateGradient(unittest.TestCase):
    def test_small_spd_system(self):
        """Solve a random 5×5 SPD system and verify the residual."""
        n = 5
        A = _make_spd(n, seed=42)
        b = _random_vector(n, seed=99)
        x = conjugate_gradient(A, b, tol=1e-10)
        res = _residual_norm(A, x, b)
        self.assertLessEqual(res, 1e-8)

    def test_identity_matrix(self):
        """CG should converge in a single iteration for the identity matrix."""
        n = 4
        A = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        b = _random_vector(n, seed=7)
        x = conjugate_gradient(A, b, tol=1e-12, max_iter=10)
        for xi, bi in zip(x, b):
            self.assertAlmostEqual(xi, bi, places=12)

    def test_max_iterations(self):
        """When max_iter is smaller than needed, CG stops early."""
        n = 8
        A = _make_spd(n, seed=123)
        b = _random_vector(n, seed=321)
        max_iter = 3
        x = conjugate_gradient(A, b, tol=1e-12, max_iter=max_iter)
        # Residual should be larger than the tight tolerance but finite.
        res = _residual_norm(A, x, b)
        self.assertGreater(res, 1e-6)

    def test_callable_operator(self):
        """Pass a linear operator instead of a dense matrix."""
        n = 6
        A_dense = _make_spd(n, seed=555)

        def A_op(v: Vector) -> Vector:
            return [sum(aij * vj for aij, vj in zip(row, v)) for row in A_dense]

        b = _random_vector(n, seed=777)
        x = conjugate_gradient(A_op, b, tol=1e-9)
        res = _residual_norm(A_dense, x, b)
        self.assertLessEqual(res, 1e-7)


def benchmark_cg():
    """Simple benchmark for increasing problem sizes."""
    print("Benchmarking Conjugate Gradient (dense SPD matrices)")
    for n in (50, 100, 200):
        A = _make_spd(n, seed=n)
        b = _random_vector(n, seed=n * 2)
        start = time.perf_counter()
        x = conjugate_gradient(A, b, tol=1e-8)
        elapsed = time.perf_counter() - start
        res = _residual_norm(A, x, b)
        print(f"n={n:3d} | time={elapsed:.4f}s | residual={res:.2e}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run a quick benchmark
    benchmark_cg()
