"""
Conjugate Gradient (CG) solver for symmetric positive‑definite (SPD) linear systems.

Provides a pure‑Python implementation using only the standard library.
All operations are performed on dense matrices represented as
list‑of‑list (row‑major) and vectors as list of floats.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

Vector = List[float]
Matrix = List[List[float]]


def _dot(u: Vector, v: Vector) -> float:
    """Return the Euclidean inner product of two vectors."""
    if len(u) != len(v):
        raise ValueError("Vectors must be of same length for dot product.")
    return sum(ui * vi for ui, vi in zip(u, v))


def _norm(v: Vector) -> float:
    """Return the Euclidean norm of a vector."""
    return _dot(v, v) ** 0.5


def _matvec(A: Matrix, x: Vector) -> Vector:
    """Matrix‑vector product y = A·x."""
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square.")
    if len(x) != n:
        raise ValueError("Dimension mismatch between A and x.")
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in A]


def _axpy(alpha: float, x: Vector, y: Vector) -> Vector:
    """Return alpha*x + y (in‑place on y)."""
    if len(x) != len(y):
        raise ValueError("Vectors must be of same length for axpy.")
    return [alpha * xi + yi for xi, yi in zip(x, y)]


def cg_solve(
    A: Matrix,
    b: Vector,
    x0: Optional[Vector] = None,
    tol: float = 1e-8,
    max_iter: Optional[int] = None,
) -> Tuple[Vector, int]:
    """
    Solve the linear system A·x = b using the Conjugate Gradient method.

    Parameters
    ----------
    A : Matrix
        Symmetric positive‑definite matrix (n×n) stored as list of rows.
    b : Vector
        Right‑hand side vector (length n).
    x0 : Vector, optional
        Initial guess. If omitted, a zero vector is used.
    tol : float, optional
        Desired relative residual norm reduction; iteration stops when
        ||r_k|| / ||b|| <= tol.
    max_iter : int, optional
        Maximum number of iterations. Defaults to n (the dimension).

    Returns
    -------
    x : Vector
        Approximate solution.
    iters : int
        Number of iterations performed.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent or if A is not square.
    """
    n = len(b)
    if any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square and match size of b.")
    if x0 is None:
        x = [0.0] * n
    else:
        if len(x0) != n:
            raise ValueError("Initial guess x0 must have same length as b.")
        x = x0[:]

    r = _axpy(-1.0, _matvec(A, x), b)          # r = b - A·x
    p = r[:]                                   # initial search direction
    rs_old = _dot(r, r)

    b_norm = _norm(b)
    if b_norm == 0.0:
        # Trivial system 0·x = 0
        return x, 0

    if max_iter is None:
        max_iter = n

    for k in range(1, max_iter + 1):
        Ap = _matvec(A, p)
        pAp = _dot(p, Ap)
        if pAp <= 0.0:
            raise ValueError("Matrix A is not positive‑definite (pᵀAp <= 0).")
        alpha = rs_old / pAp
        x = _axpy(alpha, p, x)                # x_{k+1} = x_k + α·p_k
        r = _axpy(-alpha, Ap, r)              # r_{k+1} = r_k - α·A·p_k
        rs_new = _dot(r, r)

        # Check convergence (relative residual)
        if _norm(r) / b_norm <= tol:
            return x, k

        beta = rs_new / rs_old
        p = _axpy(beta, p, r)                 # p_{k+1} = r_{k+1} + β·p_k
        rs_old = rs_new

    # Max iterations reached without meeting tolerance
    return x, max_iter


__all__ = ["cg_solve", "Matrix", "Vector"]
