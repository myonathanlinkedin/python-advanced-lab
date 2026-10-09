from __future__ import annotations
from typing import List, Tuple, Optional
from .types import Matrix, Vector, LinearSystem

def _dot(u: Vector, v: Vector) -> float:
    """Return the Euclidean inner product of two vectors."""
    if len(u) != len(v):
        raise ValueError("Vectors must be the same length for dot product.")
    return sum(ui * vi for ui, vi in zip(u, v))

def _norm(v: Vector) -> float:
    """Return the Euclidean norm of a vector."""
    return _dot(v, v) ** 0.5

def _matvec(A: Matrix, x: Vector) -> Vector:
    """Multiply matrix A by vector x (A·x)."""
    if any(len(row) != len(x) for row in A):
        raise ValueError("Incompatible dimensions for matrix‑vector multiplication.")
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in A]

def _axpy(a: float, x: Vector, y: Vector) -> Vector:
    """Return a*x + y (element‑wise)."""
    if len(x) != len(y):
        raise ValueError("Vectors must be the same length for axpy.")
    return [a * xi + yi for xi, yi in zip(x, y)]

def conjugate_gradient(
    system: LinearSystem,
    x0: Optional[Vector] = None,
    tol: float = 1e-8,
    max_iter: Optional[int] = None,
) -> Tuple[Vector, int]:
    """
    Solve Ax = b for a symmetric positive‑definite matrix A using the
    Conjugate Gradient method.

    Parameters
    ----------
    system: LinearSystem
        The linear system (A, b) to solve.
    x0: Optional[Vector]
        Initial guess. If None, a zero vector is used.
    tol: float
        Desired relative residual norm tolerance.
    max_iter: Optional[int]
        Maximum number of iterations. Defaults to the dimension of A.

    Returns
    -------
    x: Vector
        Approximate solution.
    iters: int
        Number of iterations performed.
    """
    A, b = system.A, system.b
    n = len(A)

    if max_iter is None:
        max_iter = n

    # Initial guess
    if x0 is None:
        x = [0.0] * n
    else:
        if len(x0) != n:
            raise ValueError("Initial guess x0 must have the same dimension as b.")
        x = list(x0)  # make a mutable copy

    r = _axpy(-1.0, _matvec(A, x), b)   # r = b - A·x
    p = r.copy()
    rs_old = _dot(r, r)

    if rs_old < tol * tol:
        return x, 0

    for it in range(1, max_iter + 1):
        Ap = _matvec(A, p)
        pAp = _dot(p, Ap)
        if pAp == 0.0:
            raise RuntimeError("Breakdown: pᵀ·A·p == 0, matrix may not be SPD.")
        alpha = rs_old / pAp

        # x_{k+1} = x_k + α p_k
        x = _axpy(alpha, p, x)

        # r_{k+1} = r_k - α A p_k
        r = _axpy(-alpha, Ap, r)

        rs_new = _dot(r, r)

        # Check convergence (relative residual)
        if rs_new ** 0.5 <= tol * _norm(b):
            return x, it

        beta = rs_new / rs_old
        # p_{k+1} = r_{k+1} + β p_k
        p = _axpy(beta, p, r)

        rs_old = rs_new

    # If we exit the loop, max_iter was reached without convergence
    return x, max_iter
