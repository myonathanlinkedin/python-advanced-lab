from __future__ import annotations
from typing import List
from .types import LinearSystem, Vector, Matrix
from .engine import conjugate_gradient, _dot, _norm, _matvec

def _solve_direct(system: LinearSystem) -> Vector:
    """
    Very small direct solver using Gaussian elimination with partial pivoting.
    Intended only for verification in unit tests (n <= 5).
    """
    import copy
    A = copy.deepcopy(system.A)
    b = system.b[:]
    n = len(A)

    # Forward elimination
    for k in range(n):
        # Pivot
        max_row = max(range(k, n), key=lambda i: abs(A[i][k]))
        if A[max_row][k] == 0:
            raise ValueError("Matrix is singular.")
        if max_row != k:
            A[k], A[max_row] = A[max_row], A[k]
            b[k], b[max_row] = b[max_row], b[k]

        pivot = A[k][k]
        for j in range(k, n):
            A[k][j] /= pivot
        b[k] /= pivot

        for i in range(k + 1, n):
            factor = A[i][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    # Back substitution
    x = [0.0 for _ in range(n)]
    for i in reversed(range(n)):
        x[i] = b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))
    return x

def _assert_close(v1: Vector, v2: Vector, eps: float = 1e-8) -> None:
    assert len(v1) == len(v2), "Vector length mismatch."
    for a, b in zip(v1, v2):
        assert abs(a - b) <= eps, f"Values {a} and {b} differ by more than {eps}"

def _test_known_solution() -> None:
    A: Matrix = [
        [4.0, 1.0],
        [1.0, 3.0],
    ]
    b: Vector = [1.0, 2.0]
    system = LinearSystem(A, b)
    x_cg = conjugate_gradient(system, tol=1e-12)
    # Exact solution computed analytically
    x_exact = [1.0 / 11.0, 7.0 / 11.0]
    _assert_close(x_cg, x_exact, eps=1e-9)

def _test_random_spd() -> None:
    import random
    random.seed(0)

    def generate_spd(n: int) -> Matrix:
        """Create a random symmetric positive‑definite matrix."""
        M = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
        # Form A = M^T * M + n*I to guarantee SPD
        A = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                A[i][j] = sum(M[k][i] * M[k][j] for k in range(n))
            A[i][i] += n  # shift eigenvalues
        return A

    for n in range(2, 6):
        A = generate_spd(n)
        b = [random.uniform(-5, 5) for _ in range(n)]
        system = LinearSystem(A, b)
        x_cg = conjugate_gradient(system, tol=1e-10, max_iter=2 * n)
        # Verify residual norm
        residual = _axpy(-1.0, _matvec(A, x_cg), b)  # r = b - A*x
        rel_res = _norm(residual) / _norm(b)
        assert rel_res < 1e-8, f"Relative residual {rel_res} too large for n={n}"

        # Compare against direct solver for extra confidence (n <= 5)
        x_direct = _solve_direct(system)
        _assert_close(x_cg, x_direct, eps=1e-7)

def _test_zero_rhs() -> None:
    A: Matrix = [
        [2.0, 0.0],
        [0.0, 3.0],
    ]
    b: Vector = [0.0, 0.0]
    system = LinearSystem(A, b)
    x_cg = conjugate_gradient(system, tol=1e-12)
    _assert_close(x_cg, [0.0, 0.0], eps=1e-12)

def _run_all_tests() -> None:
    _test_known_solution()
    _test_random_spd()
    _test_zero_rhs()
    print("All CG unit tests passed.")

if __name__ == "__main__":
    _run_all_tests()
    # Demonstration with a 3×3 SPD system
    A_demo: Matrix = [
        [6.0, 2.0, 1.0],
        [2.0, 5.0, 2.0],
        [1.0, 2.0, 4.0],
    ]
    b_demo: Vector = [9.0, 8.0, 7.0]
    system_demo = LinearSystem(A_demo, b_demo)
    x_solution = conjugate_gradient(system_demo, tol=1e-12)
    print("Demo solution:", x_solution)
    # Verify residual for the demo
    residual_demo = _axpy(-1.0, _matvec(A_demo, x_solution), b_demo)
    print("Demo residual norm:", _norm(residual_demo))
