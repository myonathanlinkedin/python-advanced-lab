import unittest
import random
import time
from typing import List
from core import conjugate_gradient, Vector, Matrix

def _dot(u: List[float], v: List[float]) -> float:
    return sum(a * b for a, b in zip(u, v))

def _gaussian_elimination(A: Matrix, b: Vector) -> Vector:
    """Simple dense Gaussian elimination with partial pivoting."""
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

def _random_spd_matrix(n: int) -> Matrix:
    """Generate a random symmetric positive‑definite matrix."""
    # Create a random dense matrix M
    M = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    # Compute A = Mᵀ·M + n·I to guarantee SPD
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = sum(M[k][i] * M[k][j] for k in range(n))
            if i == j:
                val += n  # shift eigenvalues
            A[i][j] = val
            A[j][i] = val
    return A

class TestConjugateGradient(unittest.TestCase):
    def test_identity(self):
        n = 4
        A = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        b = [random.uniform(-5, 5) for _ in range(n)]
        x = conjugate_gradient(A, b, tol=1e-12)
        for xi, bi in zip(x, b):
            self.assertAlmostEqual(xi, bi, places=12)

    def test_small_spd(self):
        A = [
            [4, 1, 0],
            [1, 3, 1],
            [0, 1, 2],
        ]
        b = [1, 2, 3]
        x_cg = conjugate_gradient(A, b, tol=1e-12)
        x_ref = _gaussian_elimination(A, b)
        for xc, xr in zip(x_cg, x_ref):
            self.assertAlmostEqual(xc, xr, places=10)

    def test_random_spd(self):
        n = 6
        A = _random_spd_matrix(n)
        b = [random.uniform(-10, 10) for _ in range(n)]
        x_cg = conjugate_gradient(A, b, tol=1e-10)
        x_ref = _gaussian_elimination(A, b)
        for xc, xr in zip(x_cg, x_ref):
            self.assertAlmostEqual(xc, xr, places=8)

    def test_callable_operator(self):
        n = 5
        A = _random_spd_matrix(n)
        def matvec(v: Vector) -> Vector:
            return [_dot(row, v) for row in A]
        b = [random.uniform(-3, 3) for _ in range(n)]
        x_cg = conjugate_gradient(matvec, b, tol=1e-10)
        x_ref = _gaussian_elimination(A, b)
        for xc, xr in zip(x_cg, x_ref):
            self.assertAlmostEqual(xc, xr, places=8)

    def test_non_convergent(self):
        # Use a non‑SPD matrix (e.g., indefinite) to trigger failure.
        A = [
            [0, 1],
            [1, 0],
        ]
        b = [1, 2]
        with self.assertRaises(ValueError):
            conjugate_gradient(A, b, max_iter=5, tol=1e-12)

def benchmark():
    n = 200
    A = _random_spd_matrix(n)
    b = [random.uniform(-1, 1) for _ in range(n)]
    start = time.time()
    x = conjugate_gradient(A, b, tol=1e-8)
    elapsed = time.time() - start
    residual = [_dot(row, x) - bi for row, bi in zip(A, b)]
    res_norm = math.sqrt(_dot(residual, residual))
    print(f"Benchmark: n={n}, time={elapsed:.3f}s, residual_norm={res_norm:e}")

if __name__ == "__main__":
    unittest.main(exit=False)
    print("\nDemo run with a random 5×5 SPD system:")
    demo_n = 5
    demo_A = _random_spd_matrix(demo_n)
    demo_b = [random.uniform(-5, 5) for _ in range(demo_n)]
    demo_x = conjugate_gradient(demo_A, demo_b, tol=1e-12)
    print("A =", demo_A)
    print("b =", demo_b)
    print("Solution x =", demo_x)
    # Verify solution
    residual = [_dot(row, demo_x) - bi for row, bi in zip(demo_A, demo_b)]
    print("Residual norm =", math.sqrt(_dot(residual, residual)))
    # Optional quick benchmark
    benchmark()
