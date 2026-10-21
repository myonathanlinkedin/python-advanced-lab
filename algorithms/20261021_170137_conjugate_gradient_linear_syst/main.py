from __future__ import annotations

from typing import List

from .engine import conjugate_gradient, mat_vec_mul, norm
from .types import Matrix, Vector

def _assert_close(v1: List[float], v2: List[float], eps: float = 1e-8) -> None:
    assert len(v1) == len(v2), "Vector lengths differ."
    for a, b in zip(v1, v2):
        assert abs(a - b) <= eps, f"Values {a} and {b} differ by more than {eps}"

def test_small_system() -> None:
    # A is SPD, solution known analytically: x = [0, 1]
    A: Matrix = [[4.0, 1.0],
                 [1.0, 3.0]]
    b: Vector = [1.0, 2.0]
    expected: Vector = [0.0, 1.0]

    x, iters = conjugate_gradient(A, b, tol=1e-12)
    _assert_close(x, expected, eps=1e-10)
    assert iters <= len(b), "CG should converge in at most n iterations for exact arithmetic."

def test_three_by_three() -> None:
    A: Matrix = [[6.0, 1.0, 1.0],
                 [1.0, 5.0, 2.0],
                 [1.0, 2.0, 4.0]]
    b: Vector = [7.0, 8.0, 6.0]

    x, iters = conjugate_gradient(A, b, tol=1e-10)
    # Verify residual norm is within tolerance
    residual = [bi - ai for bi, ai in zip(b, mat_vec_mul(A, x))]
    assert norm(residual) <= 1e-9 * norm(b), "Residual too large."

    # The exact solution (computed externally) is approximately:
    expected = [0.9999999999999999, 1.0000000000000002, 0.9999999999999998]
    _assert_close(x, expected, eps=1e-8)

def test_zero_rhs() -> None:
    A: Matrix = [[2.0, 0.5],
                 [0.5, 1.0]]
    b: Vector = [0.0, 0.0]
    x, iters = conjugate_gradient(A, b)
    _assert_close(x, [0.0, 0.0])
    assert iters == 0, "Zero RHS should require zero iterations."

def test_invalid_dimensions() -> None:
    A: Matrix = [[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]]
    b: Vector = [1.0, 2.0]
    try:
        conjugate_gradient(A, b)
    except ValueError as e:
        assert "square" in str(e)
    else:
        assert False, "Expected ValueError for non‑square matrix."

def test_non_spd() -> None:
    # Symmetric but not positive definite (eigenvalues: -1, 3)
    A: Matrix = [[0.0, 1.0],
                 [1.0, 0.0]]
    b: Vector = [1.0, 1.0]
    try:
        conjugate_gradient(A, b)
    except ValueError as e:
        assert "positive definite" in str(e)
    else:
        assert False, "Expected ValueError for non‑SPD matrix."

def run_all_tests() -> None:
    test_small_system()
    test_three_by_three()
    test_zero_rhs()
    test_invalid_dimensions()
    test_non_spd()
    print("All CG unit tests passed.")

if __name__ == "__main__":
    run_all_tests()
    # Demonstration with a random (but deterministic) SPD matrix
    A_demo: Matrix = [[10.0, 2.0, 3.0],
                      [2.0, 8.0, 1.0],
                      [3.0, 1.0, 5.0]]
    b_demo: Vector = [7.0, 5.0, 6.0]
    x_sol, iters = conjugate_gradient(A_demo, b_demo, tol=1e-12)
    print(f"Demo solution (iters={iters}): {x_sol}")
    # Verify residual for the demo
    residual_demo = [bi - ai for bi, ai in zip(b_demo, mat_vec_mul(A_demo, x_sol))]
    print(f"Residual norm: {norm(residual_demo):.2e}")
