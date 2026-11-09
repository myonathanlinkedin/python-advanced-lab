from __future__ import annotations
import math
from custom_types import Vector, Matrix
from engine import ConjugateGradientSolver


def assert_close(a: Vector, b: Vector, tol: float = 1e-8) -> None:
    assert len(a) == len(b), "Vector length mismatch."
    for ai, bi in zip(a, b):
        assert abs(ai - bi) <= tol, f"Values differ: {ai} vs {bi}"


def test_small_system() -> None:
    A: Matrix = [[4.0, 1.0], [1.0, 3.0]]
    b: Vector = [1.0, 2.0]
    solver = ConjugateGradientSolver(A, tol=1e-12)
    x = solver.solve(b)
    expected: Vector = [1.0 / 11.0, 7.0 / 11.0]
    assert_close(x, expected, tol=1e-10)


def test_3x3_system() -> None:
    A: Matrix = [
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ]
    b: Vector = [1.0, 2.0, 3.0]
    solver = ConjugateGradientSolver(A, tol=1e-12)
    x = solver.solve(b)
    # Verify residual norm is below tolerance
    residual = [bi - sum(A[i][j] * x[j] for j in range(3)) for i, bi in enumerate(b)]
    assert math.sqrt(sum(r * r for r in residual)) < 1e-10


def test_non_square_matrix() -> None:
    A: Matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    b: Vector = [1.0, 2.0, 3.0]
    try:
        ConjugateGradientSolver(A)
    except ValueError as e:
        assert "square" in str(e)
    else:
        assert False, "Expected ValueError for non-square matrix."


def test_non_symmetric_matrix() -> None:
    A: Matrix = [[1.0, 2.0], [3.0, 4.0]]
    b: Vector = [1.0, 2.0]
    try:
        ConjugateGradientSolver(A)
    except ValueError as e:
        assert "symmetric" in str(e)
    else:
        assert False, "Expected ValueError for non-symmetric matrix."


def test_non_positive_definite() -> None:
    A: Matrix = [[-1.0, 0.0], [0.0, 1.0]]
    b: Vector = [1.0, 2.0]
    try:
        ConjugateGradientSolver(A)
    except ValueError as e:
        assert "positive definite" in str(e)
    else:
        assert False, "Expected ValueError for non-positive-definite matrix."


def run_all_tests() -> None:
    test_small_system()
    test_3x3_system()
    test_non_square_matrix()
    test_non_symmetric_matrix()
    test_non_positive_definite()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()
    # Demo: solve a 4x4 SPD system
    A_demo: Matrix = [
        [4.0, 1.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 1.0],
        [0.0, 0.0, 1.0, 3.0],
    ]
    b_demo: Vector = [1.0, 2.0, 3.0, 4.0]
    solver_demo = ConjugateGradientSolver(A_demo, tol=1e-12)
    x_demo = solver_demo.solve(b_demo)
    print("Demo solution:", x_demo)
