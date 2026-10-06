from __future__ import annotations
import random
from custom_types import Vector, Matrix
from engine import ConjugateGradientSolver
import math


def generate_spd_matrix(n: int, seed: Optional[int] = None) -> Matrix:
    """Generate a random symmetric positive definite matrix."""
    rng = random.Random(seed)
    B = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    # Compute A = B^T * B + n * I to ensure SPD
    A = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = sum(B[k][i] * B[k][j] for k in range(n))
            A[i][j] = s
    for i in range(n):
        A[i][i] += n
    return A


def test_small_system() -> None:
    A: Matrix = [
        [4.0, 1.0],
        [1.0, 3.0]
    ]
    b: Vector = [1.0, 2.0]
    solver = ConjugateGradientSolver(A)
    x = solver.solve(b)
    # Expected solution via direct solve: x = [0.090909..., 0.636363...]
    expected = [0.09090909090909091, 0.6363636363636364]
    assert all(abs(a - b) < 1e-6 for a, b in zip(x, expected)), f"Small system test failed: {x}"


def test_random_spd() -> None:
    n = 10
    A = generate_spd_matrix(n, seed=42)
    # Create a known solution x_true
    x_true: Vector = [random.uniform(-5, 5) for _ in range(n)]
    # Compute b = A * x_true
    b: Vector = []
    for row in A:
        b.append(sum(a * x for a, x in zip(row, x_true)))
    solver = ConjugateGradientSolver(A, tol=1e-12)
    x_est = solver.solve(b)
    assert all(abs(a - b) < 1e-8 for a, b in zip(x_est, x_true)), f"Random SPD test failed: {x_est}"


def test_zero_vector() -> None:
    A: Matrix = [[2.0, 0.0], [0.0, 2.0]]
    b: Vector = [0.0, 0.0]
    solver = ConjugateGradientSolver(A)
    x = solver.solve(b)
    assert all(abs(v) < 1e-12 for v in x), f"Zero vector test failed: {x}"


def test_non_square_matrix() -> None:
    A: Matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b: Vector = [7.0, 8.0]
    try:
        ConjugateGradientSolver(A)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-square matrix did not raise ValueError.")


def test_singular_matrix() -> None:
    # Singular SPD matrix (rank deficient)
    A: Matrix = [[1.0, 2.0], [2.0, 4.0]]
    b: Vector = [3.0, 6.0]
    solver = ConjugateGradientSolver(A, max_iter=5)
    try:
        solver.solve(b)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Singular matrix did not raise RuntimeError.")


def run_tests() -> None:
    test_small_system()
    test_random_spd()
    test_zero_vector()
    test_non_square_matrix()
    test_singular_matrix()
    print("All tests passed.")


def demo() -> None:
    A: Matrix = [
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0]
    ]
    b: Vector = [1.0, 2.0, 3.0]
    solver = ConjugateGradientSolver(A)
    x = solver.solve(b)
    print("Demo solution:", x)


if __name__ == "__main__":
    run_tests()
    demo()
