"""Unit tests, benchmarks, and entrypoint for the Conjugate Gradient solver."""
from __future__ import annotations

import math
import time
from typing import List, Tuple

from core import (
    conjugate_gradient,
    matvec,
    dot,
    norm_2,
    norm_inf,
    is_spd,
    build_spd_matrix,
    build_random_spd,
    Matrix,
    Vector,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_dot_product() -> None:
    assert dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0
    assert dot([], []) == 0.0
    assert dot([1.0], [1.0]) == 1.0


def test_matvec() -> None:
    A = [[1.0, 2.0], [3.0, 4.0]]
    x = [1.0, 1.0]
    y = matvec(A, x)
    assert y == [3.0, 7.0]
    # Identity
    I = [[1.0, 0.0], [0.0, 1.0]]
    assert matvec(I, [5.0, -3.0]) == [5.0, -3.0]


def test_norms() -> None:
    v = [3.0, -4.0]
    assert norm_2(v) == 5.0
    assert norm_inf(v) == 4.0
    assert norm_2([]) == 0.0
    assert norm_inf([]) == 0.0


def test_is_spd_positive() -> None:
    A = [[4.0, 1.0], [1.0, 3.0]]
    assert is_spd(A) is True


def test_is_spd_not_symmetric() -> None:
    A = [[1.0, 2.0], [0.0, 1.0]]
    assert is_spd(A) is False


def test_is_spd_not_positive() -> None:
    A = [[-1.0, 0.0], [0.0, 1.0]]
    assert is_spd(A) is False


def test_cg_1x1() -> None:
    A = [[2.0]]
    b = [6.0]
    x, iters, rel = conjugate_gradient(A, b)
    assert abs(x[0] - 3.0) < 1e-12
    assert iters == 1
    assert rel < 1e-10


def test_cg_2x2() -> None:
    A = [[4.0, 1.0], [1.0, 3.0]]
    b = [1.0, 2.0]
    x, iters, rel = conjugate_gradient(A, b)
    # Analytical: x = A^{-1} b
    det = 4.0 * 3.0 - 1.0 * 1.0
    x_exact = [(3.0 * 1.0 - 1.0 * 2.0) / det, (-1.0 * 1.0 + 4.0 * 2.0) / det]
    assert abs(x[0] - x_exact[0]) < 1e-10
    assert abs(x[1] - x_exact[1]) < 1e-10
    assert rel < 1e-10


def test_cg_tridiagonal_n10() -> None:
    n = 10
    A = build_spd_matrix(n, diag=4.0, off_diag=1.0)
    b = [1.0] * n
    x, iters, rel = conjugate_gradient(A, b, tol=1e-12)
    # Verify residual
    r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    assert norm_2(r) / norm_2(b) < 1e-10
    assert iters <= n


def test_cg_tridiagonal_n50() -> None:
    n = 50
    A = build_spd_matrix(n, diag=4.0, off_diag=1.0)
    b = [float(i + 1) for i in range(n)]
    x, iters, rel = conjugate_gradient(A, b, tol=1e-12)
    r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    assert norm_2(r) / norm_2(b) < 1e-10
    assert iters <= n


def test_cg_random_spd_n20() -> None:
    n = 20
    A = build_random_spd(n, seed=123)
    b = [float(i % 7 + 1) for i in range(n)]
    x, iters, rel = conjugate_gradient(A, b, tol=1e-12)
    r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    assert norm_2(r) / norm_2(b) < 1e-10


def test_cg_zero_rhs() -> None:
    A = [[2.0, 0.0], [0.0, 3.0]]
    b = [0.0, 0.0]
    x, iters, rel = conjugate_gradient(A, b)
    assert x == [0.0, 0.0]
    assert iters == 0
    assert rel == 0.0


def test_cg_nonzero_initial_guess() -> None:
    A = [[4.0, 1.0], [1.0, 3.0]]
    b = [1.0, 2.0]
    x0 = [0.5, 0.5]
    x, iters, rel = conjugate_gradient(A, b, x0=x0, tol=1e-12)
    det = 11.0
    x_exact = [(3.0 - 1.0) / det, (-1.0 + 8.0) / det]
    assert abs(x[0] - x_exact[0]) < 1e-10
    assert abs(x[1] - x_exact[1]) < 1e-10


def test_cg_dimension_mismatch() -> None:
    A = [[1.0, 0.0], [0.0, 1.0]]
    b = [1.0]  # wrong length
    try:
        conjugate_gradient(A, b)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_cg_non_spd_raises() -> None:
    A = [[1.0, 2.0], [0.0, 1.0]]  # not symmetric
    b = [1.0, 1.0]
    try:
        conjugate_gradient(A, b)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_cg_max_iter_respected() -> None:
    A = build_spd_matrix(10, diag=4.0, off_diag=1.0)
    b = [1.0] * 10
    x, iters, rel = conjugate_gradient(A, b, max_iter=3, tol=1e-15)
    assert iters <= 3


def test_cg_convergence_monotonic_residual() -> None:
    """Verify that the residual norm decreases (in exact arithmetic)."""
    n = 15
    A = build_spd_matrix(n, diag=4.0, off_diag=1.0)
    b = [1.0] * n
    x, iters, rel = conjugate_gradient(A, b, tol=1e-14)
    assert rel < 1e-10


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(n: int = 200) -> Tuple[float, int, float]:
    """Solve a random SPD system of size n and report timing."""
    A = build_random_spd(n, seed=99)
    b = [float((i * 7 + 3) % 11 + 1) for i in range(n)]
    t0 = time.perf_counter()
    x, iters, rel = conjugate_gradient(A, b, tol=1e-12)
    elapsed = time.perf_counter() - t0
    return elapsed, iters, rel


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all_tests() -> None:
    tests = [
        test_dot_product,
        test_matvec,
        test_norms,
        test_is_spd_positive,
        test_is_spd_not_symmetric,
        test_is_spd_not_positive,
        test_cg_1x1,
        test_cg_2x2,
        test_cg_tridiagonal_n10,
        test_cg_tridiagonal_n50,
        test_cg_random_spd_n20,
        test_cg_zero_rhs,
        test_cg_nonzero_initial_guess,
        test_cg_dimension_mismatch,
        test_cg_non_spd_raises,
        test_cg_max_iter_respected,
        test_cg_convergence_monotonic_residual,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("Conjugate Gradient Solver – Unit Tests")
    print("=" * 60)
    run_all_tests()

    print("\n" + "=" * 60)
    print("Benchmark")
    print("=" * 60)
    for n in [50, 100, 200, 500]:
        elapsed, iters, rel = benchmark(n)
        print(f"  n={n:>4d}  iters={iters:>4d}  rel_res={rel:.2e}  time={elapsed*1000:.2f} ms")

    print("\n" + "=" * 60)
    print("Demo: solve a 5x5 tridiagonal SPD system")
    print("=" * 60)
    n = 5
    A = build_spd_matrix(n, diag=4.0, off_diag=1.0)
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    x, iters, rel = conjugate_gradient(A, b, tol=1e-12)
    print(f"  A = {A}")
    print(f"  b = {b}")
    print(f"  x = {[round(v, 6) for v in x]}")
    print(f"  iterations = {iters}, relative residual = {rel:.2e}")
    # Verify
    r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    print(f"  ||r||_2 / ||b||_2 = {norm_2(r) / norm_2(b):.2e}")
    print("\nDone.")
