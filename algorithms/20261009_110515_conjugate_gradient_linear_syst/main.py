from __future__ import annotations

import math
import sys
from typing import List

from .engine import conjugate_gradient
from .types import Matrix, Vector

def _assert_close(v1: Vector, v2: Vector, eps: float = 1e-6) -> None:
    if len(v1) != len(v2):
        raise AssertionError("Vector lengths differ.")
    for a, b in zip(v1, v2):
        if not math.isclose(a, b, rel_tol=eps, abs_tol=eps):
            raise AssertionError(f"Values {a} and {b} differ beyond tolerance {eps}.")

def _run_unit_tests() -> None:
    # Test 1: 2×2 SPD matrix with known solution.
    A1: Matrix = [
        [4.0, 1.0],
        [1.0, 3.0],
    ]
    x_true1: Vector = [1.0, 2.0]
    b1: Vector = [4.0 * x_true1[0] + 1.0 * x_true1[1],
                  1.0 * x_true1[0] + 3.0 * x_true1[1]]
    x_computed1 = conjugate_gradient(A1, b1, tol=1e-10)
    _assert_close(x_computed1, x_true1)

    # Test 2: 3×3 SPD matrix (Hilbert matrix is SPD but ill‑conditioned).
    A2: Matrix = [
        [1.0, 0.5, 1.0/3.0],
        [0.5, 1.0/3.0, 0.25],
        [1.0/3.0, 0.25, 0.2],
    ]
    x_true2: Vector = [1.0, -1.0, 2.0]
    b2: Vector = [sum(A2[i][j] * x_true2[j] for j in range(3)) for i in range(3)]
    x_computed2 = conjugate_gradient(A2, b2, tol=1e-12, max_iter=1000)
    _assert_close(x_computed2, x_true2, eps=1e-5)

    # Test 3: Zero right‑hand side should yield zero solution regardless of initial guess.
    A3: Matrix = [
        [2.0, 0.0],
        [0.0, 5.0],
    ]
    b3: Vector = [0.0, 0.0]
    x0_guess: Vector = [7.0, -3.0]
    x_computed3 = conjugate_gradient(A3, b3, x0=x0_guess, tol=1e-12)
    _assert_close(x_computed3, [0.0, 0.0])

    # Test 4: Dimension mismatch raises ValueError.
    try:
        conjugate_gradient([[1.0, 2.0]], [1.0, 2.0, 3.0])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for dimension mismatch.")

    # Test 5: Non‑square matrix raises ValueError.
    try:
        conjugate_gradient([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [1.0, 2.0, 3.0])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non‑square matrix.")

    print("All unit tests passed.")

def _demo() -> None:
    """
    Demonstrate solving a simple 4×4 SPD system.
    The matrix is constructed as A = Mᵀ·M + λ·I to guarantee SPD.
    """
    M: Matrix = [
        [1.0, 2.0, 0.0, 1.0],
        [0.0, 1.0, 3.0, 2.0],
        [4.0, 0.0, 1.0, 0.0],
    ]
    # Compute A = Mᵀ·M + 0.1·I
    def transpose(mat: Matrix) -> Matrix:
        return [list(col) for col in zip(*mat)]

    def matmul(A: Matrix, B: Matrix) -> Matrix:
        n, m = len(A), len(B[0])
        p = len(B)
        return [[sum(A[i][k] * B[k][j] for k in range(p)) for j in range(m)] for i in range(n)]

    MT = transpose(M)
    A = matmul(MT, M)
    n = len(A)
    for i in range(n):
        A[i][i] += 0.1  # regularization term

    x_true: Vector = [1.5, -2.0, 0.5, 3.0]
    b = [sum(A[i][j] * x_true[j] for j in range(n)) for i in range(n)]

    x_sol = conjugate_gradient(A, b, tol=1e-10)
    print("True solution :", x_true)
    print("Computed solution:", [round(v, 6) for v in x_sol])

if __name__ == "__main__":
    _run_unit_tests()
    _demo()
