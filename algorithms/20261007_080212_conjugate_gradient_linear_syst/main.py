import unittest
import random
import time
from typing import List

from core import cg, matvec, dot, norm

class TestConjugateGradient(unittest.TestCase):
    def test_small_system(self):
        A = [
            [4.0, 1.0],
            [1.0, 3.0],
        ]
        b = [1.0, 2.0]
        expected = [0.09090909090909091, 0.6363636363636364]
        x = cg(A, b, tol=1e-12)
        for xi, ei in zip(x, expected):
            self.assertAlmostEqual(xi, ei, places=7)

    def test_random_spd(self):
        n = 10
        M = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
        # Construct SPD matrix A = M^T M + epsilon I
        epsilon = 0.1
        A = [[sum(M[k][i] * M[k][j] for k in range(n)) + (epsilon if i == j else 0.0)
              for j in range(n)] for i in range(n)]
        b = [random.uniform(-1, 1) for _ in range(n)]
        x = cg(A, b, tol=1e-10)
        # Verify Ax ≈ b
        Ax = matvec(A, x)
        for ai, bi in zip(Ax, b):
            self.assertAlmostEqual(ai, bi, places=6)

    def test_convergence_rate(self):
        n = 50
        M = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
        epsilon = 0.5
        A = [[sum(M[k][i] * M[k][j] for k in range(n)) + (epsilon if i == j else 0.0)
              for j in range(n)] for i in range(n)]
        b = [random.uniform(-1, 1) for _ in range(n)]
        x0 = [0.0] * n
        # Run with a very tight tolerance
        x = cg(A, b, x0=x0, tol=1e-12, max_iter=200)
        # Check residual norm
        r = [bi - sum(A[i][j] * x[j] for j in range(n)) for i, bi in enumerate(b)]
        self.assertLess(norm(r), 1e-12)

def benchmark():
    n = 200
    random.seed(0)
    M = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    epsilon = 0.1
    A = [[sum(M[k][i] * M[k][j] for k in range(n)) + (epsilon if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    b = [random.uniform(-1, 1) for _ in range(n)]
    start = time.time()
    x = cg(A, b, tol=1e-8)
    elapsed = time.time() - start
    print(f"CG solved 200x200 system in {elapsed:.4f} seconds")

if __name__ == "__main__":
    print("Running unit tests...")
    unittest.main(exit=False)
    print("\nRunning benchmark...")
    benchmark()
    print("\nDemo: solving a 3x3 system")
    A_demo = [
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ]
    b_demo = [1.0, 2.0, 3.0]
    x_demo = cg(A_demo, b_demo, tol=1e-12)
    print("Solution:", x_demo)
