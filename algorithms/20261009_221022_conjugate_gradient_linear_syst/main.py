from __future__ import annotations
import random
import math
from typing import List
from .types import LinearSystem, Matrix, Vector
from .engine import conjugate_gradient

def _is_spd(A: Matrix) -> bool:
    """Very naive SPD check using Sylvester's criterion for small matrices."""
    n = len(A)
    # All leading principal minors must be positive
    for k in range(1, n + 1):
        # Build k×k leading principal submatrix
        sub = [row[:k] for row in A[:k]]
        # Compute determinant via Laplace expansion (acceptable for tiny k)
        det = _determinant(sub)
        if det <= 0:
            return False
    return True

def _determinant(M: Matrix) -> float:
    """Recursive determinant (acceptable only for very small matrices)."""
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    det = 0.0
    for col in range(n):
        sign = (-1) ** col
        minor = [row[:col] + row[col + 1 :] for row in M[1:]]
        det += sign * M[0][col] * _determinant(minor)
    return det

def _generate_spd_matrix(n: int, seed: int = 0) -> Matrix:
    """Generate a deterministic symmetric positive‑definite matrix of size n."""
    random.seed(seed)
    # Start with a random matrix R and form A = Rᵀ·R + n·I (ensures SPD)
    R: Matrix = [[random.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]
    # Compute Rᵀ·R
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = sum(R[k][i] * R[k][j] for k in range(n))
            A[i][j] = val
            A[j][i] = val
    # Add n*I to improve conditioning
    for i in range(n):
        A[i][i] += n
    assert _is_spd(A), "Generated matrix failed SPD check."
    return A

def _test_basic_2x2() -> None:
    A = [[4.0, 1.0],
         [1.0, 3.0]]
    b = [1.0, 2.0]
    system = LinearSystem(A, b)
    x, iters = conjugate_gradient(system, tol=1e-12)
    # Expected solution from manual calculation: x = [0.090909..., 0.636363...]
    expected = [0.09090909090909091, 0.6363636363636364]
    for xi, ei in zip(x, expected):
        assert math.isclose(xi, ei, rel_tol=1e-9, abs_tol=1e-12), f"2x2 test failed: {x}"
    assert iters <= 2, "CG should converge in at most n iterations for exact arithmetic."

def _test_random_spd() -> None:
    n = 5
    A = _generate_spd_matrix(n, seed=42)
    # Choose a known solution x_true and compute b = A·x_true
    x_true = [float(i + 1) for i in range(n)]  # [1,2,3,4,5]
    b = [sum(A[i][j] * x_true[j] for j in range(n)) for i in range(n)]
    system = LinearSystem(A, b)
    x_approx, iters = conjugate_gradient(system, tol=1e-10)
    # Verify residual norm
    residual = [b[i] - sum(A[i][j] * x_approx[j] for j in range(n)) for i in range(n)]
    res_norm = math.sqrt(sum(r * r for r in residual))
    b_norm = math.sqrt(sum(bi * bi for bi in b))
    assert res_norm <= 1e-9 * b_norm, f"Residual too large: {res_norm}"
    # Verify each component close to true solution
    for xa, xt in zip(x_approx, x_true):
        assert math.isclose(xa, xt, rel_tol=1e-8, abs_tol=1e-10), f"Component mismatch: {xa} vs {xt}"
    assert iters <= n, "CG should converge within n iterations for exact arithmetic."

def _test_invalid_dimensions() -> None:
    A = [[1.0, 2.0], [3.0, 4.0]]
    b = [1.0]  # mismatched length
    try:
        LinearSystem(A, b)
    except ValueError:
        pass
    else:
        assert False, "LinearSystem should raise on dimension mismatch."

    # Non‑square matrix
    A2 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b2 = [1.0, 2.0]
    try:
        LinearSystem(A2, b2)
    except ValueError:
        pass
    else:
        assert False, "LinearSystem should raise on non‑square matrix."

def _test_zero_initial_guess_convergence() -> None:
    # 1×1 SPD matrix
    A = [[2.0]]
    b = [8.0]
    system = LinearSystem(A, b)
    x, iters = conjugate_gradient(system, x0=[0.0], tol=1e-12)
    assert math.isclose(x[0], 4.0, rel_tol=1e-12), "1x1 solution incorrect."
    assert iters == 1, "1x1 CG should finish in a single iteration."

def run_all_tests() -> None:
    _test_basic_2x2()
    _test_random_spd()
    _test_invalid_dimensions()
    _test_zero_initial_guess_convergence()
    print("All CG unit tests passed.")

if __name__ == "__main__":
    run_all_tests()
    # Demonstration with a 3×3 system
    A_demo = [[6.0, 2.0, 1.0],
              [2.0, 5.0, 2.0],
              [1.0, 2.0, 4.0]]
    b_demo = [9.0, 8.0, 7.0]
    system_demo = LinearSystem(A_demo, b_demo)
    solution, iters = conjugate_gradient(system_demo, tol=1e-10)
    print(f"Demo solution (iters={iters}): {solution}")
