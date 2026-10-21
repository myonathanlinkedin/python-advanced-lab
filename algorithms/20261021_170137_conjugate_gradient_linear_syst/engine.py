from __future__ import annotations

from typing import Optional, Tuple
from .types import Matrix, Vector

def _validate_dimensions(A: Matrix, b: Vector) -> None:
    n = len(A)
    if n == 0:
        raise ValueError("Matrix A must not be empty.")
    if any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square.")
    if len(b) != n:
        raise ValueError("Vector b length must match dimensions of A.")

def _is_symmetric(A: Matrix) -> bool:
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] != A[j][i]:
                return False
    return True

def _is_positive_definite(A: Matrix) -> bool:
    # Simple (non‑exhaustive) check: all leading principal minors must be > 0.
    # For small matrices this is sufficient for the test suite.
    n = len(A)
    for k in range(1, n + 1):
        # Build k×k leading principal sub‑matrix
        sub = [row[:k] for row in A[:k]]
        # Compute its determinant via Laplace expansion (recursive, acceptable for tiny k)
        if _determinant(sub) <= 0:
            return False
    return True

def _determinant(M: Matrix) -> float:
    # Recursive determinant for small matrices (size ≤ 4)
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    det = 0.0
    for col in range(n):
        sign = -1.0 if col % 2 else 1.0
        minor = [row[:col] + row[col + 1 :] for row in M[1:]]
        det += sign * M[0][col] * _determinant(minor)
    return det

def mat_vec_mul(A: Matrix, x: Vector) -> Vector:
    """Return the product A·x."""
    _validate_dimensions(A, x)  # type: ignore[arg-type] – dimensions must match
    return [sum(a_ij * x_j for a_ij, x_j in zip(row, x)) for row in A]

def dot(u: Vector, v: Vector) -> float:
    """Euclidean inner product."""
    if len(u) != len(v):
        raise ValueError("Vectors must be of same length for dot product.")
    return sum(ui * vi for ui, vi in zip(u, v))

def norm(v: Vector) -> float:
    """Euclidean norm (L2)."""
    return dot(v, v) ** 0.5

def conjugate_gradient(
    A: Matrix,
    b: Vector,
    x0: Optional[Vector] = None,
    tol: float = 1e-10,
    max_iter: Optional[int] = None,
) -> Tuple[Vector, int]:
    """
    Solve the symmetric positive‑definite linear system A·x = b using the
    Conjugate Gradient method.

    Parameters
    ----------
    A : Matrix
        Symmetric positive‑definite matrix.
    b : Vector
        Right‑hand side vector.
    x0 : Vector, optional
        Initial guess (defaults to zero vector).
    tol : float, optional
        Desired relative residual tolerance.
    max_iter : int, optional
        Maximum number of iterations (defaults to len(b)).

    Returns
    -------
    x : Vector
        Approximate solution.
    iters : int
        Number of iterations performed.
    """
    _validate_dimensions(A, b)

    if not _is_symmetric(A):
        raise ValueError("Matrix A must be symmetric.")
    if not _is_positive_definite(A):
        raise ValueError("Matrix A must be positive definite.")

    n = len(b)
    if x0 is None:
        x = [0.0] * n
    else:
        if len(x0) != n:
            raise ValueError("Initial guess x0 must have same length as b.")
        x = x0[:]

    r = [bi - ai for bi, ai in zip(b, mat_vec_mul(A, x))]
    p = r[:]
    rs_old = dot(r, r)

    if rs_old == 0.0:
        return x, 0  # already solved

    if max_iter is None:
        max_iter = n

    for it in range(1, max_iter + 1):
        Ap = mat_vec_mul(A, p)
        pAp = dot(p, Ap)
        if pAp == 0.0:
            raise RuntimeError("Breakdown: pᵀ·A·p == 0")
        alpha = rs_old / pAp

        # x_{k+1} = x_k + α·p_k
        x = [xi + alpha * pi for xi, pi in zip(x, p)]

        # r_{k+1} = r_k - α·A·p_k
        r = [ri - alpha * api for ri, api in zip(r, Ap)]
        rs_new = dot(r, r)

        # Check convergence (relative residual)
        if rs_new ** 0.5 <= tol * norm(b):
            return x, it

        beta = rs_new / rs_old
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rs_old = rs_new

    return x, max_iter
