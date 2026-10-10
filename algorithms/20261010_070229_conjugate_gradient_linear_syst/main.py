from __future__ import annotations
import math
from typing import List

from .engine import conjugate_gradient, CGResult
from .types import Matrix, Vector

def _almost_equal(v1: List[float], v2: List[float], *, eps: float = 1e-6) -> bool:
    """Return True if two vectors are element‑wise equal within eps."""
    if len(v1) != len(v2):
        return False
    return all(abs(a - b) <= eps for a, b in zip(v1, v2))

def _run_basic_test() -> None:
    """
    Solve a 2×2 SPD system with a known analytical solution.
    A = [[4, 1],
         [1, 3]]
    b = [1, 2]
    Exact solution: x = [1/11, 7/11] ≈ [0.090909..., 0.636363...]
    """
    A: Matrix = [[4.0, 1.0],
                 [1.0, 3.0]]
    b: Vector = [1.0, 2.0]
    expected: Vector = [1.0 / 11.0, 7.0 / 11.0]

    result: CGResult = conjugate_gradient(A, b, tol=1e-12, max_iter=100)

    assert result.converged, "CG did not converge on the basic test."
    assert result.iterations <= len(A), "CG exceeded theoretical iteration bound."
    assert _almost_equal(result.x, expected, eps=1e-8), f"Solution mismatch: {result.x} vs {expected}"
    # Verify residual norm is within tolerance
    residual = [bi - sum(Ai[j] * result.x[j] for j in range(len(A))) for i, bi in enumerate(b)]
    residual_norm = math.sqrt(sum(r * r for r in residual))
    assert residual_norm / math.sqrt(sum(bi * bi for bi in b)) <= 1e-12, "Residual too large."

def _run_zero_rhs_test() -> None:
    """If b is the zero vector, the solution must be the zero vector regardless of A."""
    A: Matrix = [[2.0, 0.5],
                 [0.5, 1.0]]
    b: Vector = [0.0, 0.0]
    result = conjugate_gradient(A, b, tol=1e-10)
    assert result.converged, "CG failed to converge on zero RHS."
    assert _almost_equal(result.x, [0.0, 0.0]), "Non‑zero solution for zero RHS."

def _run_invalid_input_test() -> None:
    """Check that malformed inputs raise appropriate exceptions."""
    A: Matrix = [[1.0, 2.0],
                 [3.0, 4.0]]
    b: Vector = [1.0, 2.0, 3.0]  # mismatched dimension
    try:
        conjugate_gradient(A, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for dimension mismatch.")

    # Non‑square matrix
    A_rect: Matrix = [[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]]
    b_rect: Vector = [1.0, 2.0]
    try:
        conjugate_gradient(A_rect, b_rect)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non‑square matrix.")

def _run_random_spd_test() -> None:
    """
    Generate a small random symmetric positive‑definite matrix and verify that
    CG reduces the residual below the tolerance. The exact solution is not
    known analytically, but we can check the residual.
    """
    import random
    random.seed(0)

    n = 5
    # Build a random matrix M and set A = Mᵀ·M + n·I (guaranteed SPD)
    M: Matrix = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    # Compute A = Mᵀ·M
    A: Matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = sum(M[k][i] * M[k][j] for k in range(n))
    # Add n*I to improve conditioning
    for i in range(n):
        A[i][i] += n

    # Random RHS
    b: Vector = [random.uniform(-1, 1) for _ in range(n)]

    result = conjugate_gradient(A, b, tol=1e-9, max_iter=2 * n)

    assert result.converged, "CG failed to converge on random SPD matrix."
    # Compute residual to double‑check
    Ax = [sum(A[i][j] * result.x[j] for j in range(n)) for i in range(n)]
    residual = [b[i] - Ax[i] for i in range(n)]
    residual_norm = math.sqrt(sum(r * r for r in residual))
    b_norm = math.sqrt(sum(bi * bi for bi in b))
    assert residual_norm / b_norm <= 1e-9, "Residual not within tolerance for random SPD test."

def _demo() -> None:
    """Simple interactive demonstration."""
    A: Matrix = [[4.0, 1.0],
                 [1.0, 3.0]]
    b: Vector = [1.0, 2.0]
    print("Solving A·x = b with Conjugate Gradient")
    print(f"A = {A}")
    print(f"b = {b}")
    result = conjugate_gradient(A, b, tol=1e-12)
    print("Result:", result)
    print("Approximate solution x:", result.x)

if __name__ == "__main__":
    _run_basic_test()
    _run_zero_rhs_test()
    _run_invalid_input_test()
    _run_random_spd_test()
    _demo()
    print("All tests passed.")
