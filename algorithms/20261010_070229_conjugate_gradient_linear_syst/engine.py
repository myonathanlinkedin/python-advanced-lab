from __future__ import annotations
from typing import List, Sequence, Tuple, Callable, Optional
import math

from .types import Vector, Matrix, MatVec, CGResult

def _dot(u: Vector, v: Vector) -> float:
    """Return the Euclidean inner product of two vectors."""
    if len(u) != len(v):
        raise ValueError("Vectors must be of the same length for dot product.")
    return sum(au * av for au, av in zip(u, v))

def _norm(v: Vector) -> float:
    """Return the Euclidean norm (L2) of a vector."""
    return math.sqrt(_dot(v, v))

def _add(u: Vector, v: Vector) -> Vector:
    """Element‑wise addition of two vectors."""
    if len(u) != len(v):
        raise ValueError("Vectors must be of the same length for addition.")
    return [au + av for au, av in zip(u, v)]

def _sub(u: Vector, v: Vector) -> Vector:
    """Element‑wise subtraction (u - v)."""
    if len(u) != len(v):
        raise ValueError("Vectors must be of the same length for subtraction.")
    return [au - av for au, av in zip(u, v)]

def _scale(alpha: float, v: Vector) -> Vector:
    """Multiply a vector by a scalar."""
    return [alpha * av for av in v]

def _matvec(A: Matrix, x: Vector) -> Vector:
    """Multiply a dense matrix A by a vector x (A·x)."""
    if not A:
        raise ValueError("Matrix A must not be empty.")
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square.")
    if len(x) != n:
        raise ValueError("Dimension mismatch between A and vector x.")
    return [_dot(row, x) for row in A]

def conjugate_gradient(
    A: Matrix | MatVec,
    b: Vector,
    *,
    x0: Optional[Vector] = None,
    tol: float = 1e-8,
    max_iter: Optional[int] = None,
) -> CGResult:
    """
    Solve the symmetric positive‑definite linear system A·x = b using the
    Conjugate Gradient method.

    Parameters
    ----------
    A : Matrix or callable
        Either a dense square matrix (list‑of‑lists) or a function that
        implements the matrix‑vector product A·v.
    b : Vector
        Right‑hand side vector.
    x0 : Vector, optional
        Initial guess. If omitted, a zero vector of appropriate size is used.
    tol : float, default 1e-8
        Desired relative residual tolerance. Iteration stops when
        ||r_k||_2 / ||b||_2 <= tol.
    max_iter : int, optional
        Upper bound on iterations. Defaults to the dimension of A.

    Returns
    -------
    CGResult
        Container with the approximate solution and diagnostic information.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent or if A is not square.
    """
    # Resolve matrix‑vector product
    if callable(A):
        matvec: MatVec = A
        # Infer dimension from b (cannot verify square property)
        n = len(b)
    else:
        # Dense matrix case
        n = len(A)
        if any(len(row) != n for row in A):
            raise ValueError("Matrix A must be square.")
        matvec = lambda v: _matvec(A, v)

    if len(b) != n:
        raise ValueError("Dimension mismatch between A and b.")

    if x0 is None:
        x = [0.0] * n
    else:
        if len(x0) != n:
            raise ValueError("Initial guess x0 has incorrect dimension.")
        x = list(x0)  # make a mutable copy

    r = _sub(b, matvec(x))          # residual r0 = b - A·x0
    p = r.copy()                    # initial search direction
    rs_old = _dot(r, r)

    b_norm = _norm(b)
    if b_norm == 0.0:
        # Trivial system 0·x = 0; return zero vector immediately.
        return CGResult(x=[0.0] * n, iterations=0, residual_norm=0.0, converged=True)

    if max_iter is None:
        max_iter = n

    converged = False
    for k in range(1, max_iter + 1):
        Ap = matvec(p)
        pAp = _dot(p, Ap)
        if pAp <= 0.0:
            # For SPD matrices pᵀAp must be positive; a non‑positive value indicates
            # either loss of SPD property or numerical breakdown.
            raise ValueError("Breakdown: non‑positive pᵀAp encountered; matrix may not be SPD.")
        alpha = rs_old / pAp
        x = _add(x, _scale(alpha, p))
        r = _sub(r, _scale(alpha, Ap))
        rs_new = _dot(r, r)

        residual_norm = math.sqrt(rs_new)
        if residual_norm / b_norm <= tol:
            converged = True
            break

        beta = rs_new / rs_old
        p = _add(r, _scale(beta, p))
        rs_old = rs_new

    return CGResult(
        x=x,
        iterations=k,
        residual_norm=residual_norm,
        converged=converged,
    )
