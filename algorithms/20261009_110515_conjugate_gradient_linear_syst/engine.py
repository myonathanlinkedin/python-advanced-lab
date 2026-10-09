from __future__ import annotations

import math
from typing import List, Optional

from .types import Matrix, Vector

def _dot(u: Vector, v: Vector) -> float:
    """Return the Euclidean dot product of two vectors."""
    if len(u) != len(v):
        raise ValueError("Vectors must be of same length for dot product.")
    return sum(au * av for au, av in zip(u, v))

def _matvec(A: Matrix, x: Vector) -> Vector:
    """Multiply matrix A by vector x (A·x)."""
    if not A:
        raise ValueError("Matrix A must not be empty.")
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square.")
    if len(x) != n:
        raise ValueError("Dimension mismatch between A and vector x.")
    return [_dot(row, x) for row in A]

def _vec_add(u: Vector, v: Vector) -> Vector:
    """Element‑wise addition of two vectors."""
    if len(u) != len(v):
        raise ValueError("Vectors must be of same length for addition.")
    return [au + av for au, av in zip(u, v)]

def _vec_sub(u: Vector, v: Vector) -> Vector:
    """Element‑wise subtraction of two vectors (u - v)."""
    if len(u) != len(v):
        raise ValueError("Vectors must be of same length for subtraction.")
    return [au - av for au, av in zip(u, v)]

def _scalar_mul(alpha: float, v: Vector) -> Vector:
    """Multiply vector v by scalar alpha."""
    return [alpha * av for av in v]

def conjugate_gradient(
    A: Matrix,
    b: Vector,
    x0: Optional[Vector] = None,
    tol: float = 1e-8,
    max_iter: Optional[int] = None,
) -> Vector:
    """
    Solve the linear system A·x = b for a symmetric positive‑definite matrix A
    using the Conjugate Gradient method.

    Parameters
    ----------
    A : Matrix
        Square, symmetric, positive‑definite matrix.
    b : Vector
        Right‑hand side vector.
    x0 : Vector, optional
        Initial guess. If omitted, a zero vector is used.
    tol : float, optional
        Desired absolute tolerance on the residual norm (default 1e‑8).
    max_iter : int, optional
        Maximum number of iterations. If omitted, defaults to the dimension of A.

    Returns
    -------
    x : Vector
        Approximate solution to A·x = b.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent.
    """
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square.")
    if len(b) != n:
        raise ValueError("Dimension mismatch between A and b.")

    if max_iter is None:
        max_iter = n * 10  # generous upper bound

    # Initial guess
    x: Vector = x0[:] if x0 is not None else [0.0] * n
    if len(x) != n:
        raise ValueError("Initial guess x0 has incorrect dimension.")

    # r = b - A·x
    r = _vec_sub(b, _matvec(A, x))
    p = r[:]  # initial search direction
    rsold = _dot(r, r)

    if math.isclose(rsold, 0.0, abs_tol=tol):
        # The initial guess already satisfies the system.
        return x

    for _ in range(max_iter):
        Ap = _matvec(A, p)
        pAp = _dot(p, Ap)
        if pAp == 0.0:
            raise RuntimeError("Breakdown: pᵀ·A·p is zero.")
        alpha = rsold / pAp

        x = _vec_add(x, _scalar_mul(alpha, p))
        r = _vec_sub(r, _scalar_mul(alpha, Ap))

        rsnew = _dot(r, r)
        if math.sqrt(rsnew) < tol:
            break

        beta = rsnew / rsold
        p = _vec_add(r, _scalar_mul(beta, p))
        rsold = rsnew

    return x

__all__: List[str] = ["conjugate_gradient"]
