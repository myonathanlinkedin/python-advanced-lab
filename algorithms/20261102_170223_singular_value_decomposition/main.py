from __future__ import annotations
import math
from typing import List
from .engine import svd, low_rank_approx
from .types import Matrix, SVDResult

def matrix_close(a: Matrix, b: Matrix, tol: float = 1e-8) -> bool:
    if len(a) != len(b) or any(len(r1) != len(r2) for r1, r2 in zip(a, b)):
        return False
    for row_a, row_b in zip(a, b):
        for x, y in zip(row_a, row_b):
            if abs(x - y) > tol:
                return False
    return True

def test_svd_reconstruction():
    A: Matrix = [
        [3.0, 1.0, 1.0],
        [-1.0, 3.0, 1.0]
    ]  # 2 x 3 matrix
    res: SVDResult = svd(A)
    # Reconstruct A = U * Σ * Vᵀ
    # Build Σ matrix (full size)
    m, n = len(A), len(A[0])
    r = len(res.s)
    Sigma: Matrix = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(r):
        Sigma[i][i] = res.s[i]
    US = [[sum(res.u[row][k] * Sigma[k][col] for k in range(r)) for col in range(n)] for row in range(m)]
    recon = [[sum(US[i][k] * res.vt[k][j] for k in range(r)) for j in range(n)] for i in range(m)]
    assert matrix_close(A, recon, tol=1e-6), "Full SVD reconstruction failed"

def test_low_rank_approx():
    A: Matrix = [
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5]
    ]  # Diagonal 3x3 matrix
    # Rank‑1 approximation should keep the largest singular value (2.0)
    approx = low_rank_approx(A, k=1)
    expected: Matrix = [
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]
    assert matrix_close(approx, expected, tol=1e-6), "Rank‑1 approximation incorrect"

def test_zero_matrix():
    A: Matrix = [
        [0.0, 0.0],
        [0.0, 0.0]
    ]
    res = svd(A)
    assert all(abs(s) < 1e-12 for s in res.s), "Singular values of zero matrix should be zero"
    approx = low_rank_approx(A, k=2)
    assert matrix_close(approx, A), "Approximation of zero matrix failed"

def test_rectangular_more_rows():
    A: Matrix = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]  # 3 x 2
    res = svd(A)
    # Verify dimensions
    assert len(res.u) == 3
    assert len(res.vt) == len(res.s)
    assert len(res.vt[0]) == 2
    # Reconstruction test
    m, n = 3, 2
    r = len(res.s)
    Sigma = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(r):
        Sigma[i][i] = res.s[i]
    US = [[sum(res.u[row][k] * Sigma[k][col] for k in range(r)) for col in range(n)] for row in range(m)]
    recon = [[sum(US[i][k] * res.vt[k][j] for k in range(r)) for j in range(n)] for i in range(m)]
    assert matrix_close(A, recon, tol=1e-6), "Rectangular SVD reconstruction failed"

def run_all_tests():
    test_svd_reconstruction()
    test_low_rank_approx()
    test_zero_matrix()
    test_rectangular_more_rows()
    print("All unit tests passed.")

if __name__ == '__main__':
    run_all_tests()
    # Demo: low‑rank approximation of a random (small) matrix
    demo_matrix: Matrix = [
        [4.0, 0.0, 2.0, 2.0],
        [0.0, 3.0, 0.0, 1.0],
        [2.0, 0.0, 3.0, 0.0],
        [2.0, 1.0, 0.0, 2.0]
    ]
    k = 2
    approx = low_rank_approx(demo_matrix, k)
    print(f"\nOriginal matrix ({len(demo_matrix)}x{len(demo_matrix[0])}):")
    for row in demo_matrix:
        print(row)
    print(f"\nRank‑{k} approximation:")
    for row in approx:
        print(row)
