from __future__ import annotations
from typing import List, Tuple, Optional
from .types import Matrix, Vector, LinearSystem

def _dot(u: Vector, v: Vector) -> float:
    """Return the Euclidean inner product of two vectors."""
    if len(u) != len(v):
        raise ValueError("Vectors must be of same length for dot product.")
    return sum(ua * va for ua, va in zip(u, v))

def _norm(v: Vector) -> float:
    """Return the Euclidean norm of a vector."""
    return _dot(v, v) ** 0.5

def _matvec(A: Matrix, x: Vector) -> Vector:
    """Matrix‑vector multiplication y = A·x."""
    n = len(A)
    if len(x) != n:
        raise ValueError("Dimension mismatch in matvec.")
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]

def _axpy(alpha: float, x: Vector, y: Vector) -> Vector:
    """Return alpha*x + y (in‑place safe)."""
    if len(x) != len(y):
        raise ValueError("Dimension mismatch in axpy.")
    return [alpha * xi + yi for xi, yi in zip(x, y)]

def _scalar_mul(alpha: float, x: Vector) -> Vector:
    """Return alpha * x."""
    return [alpha * xi for xi in x]

def conjugate_gradient(
    system: LinearSystem,
    x0: Optional[Vector] = None,
    tol: float = 1e-10,
    max_iter: Optional[int] = None,
) -> Vector:
    """
    Solve Ax = b for a symmetric positive‑definite matrix A using the
    Conjugate Gradient method.

    Parameters
    ----------
    system: LinearSystem
        The linear system to solve.
    x0: Optional[Vector]
        Initial guess. If None, a zero vector is used.
    tol: float
        Desired relative residual norm reduction.
    max_iter: Optional[int]
        Maximum number of iterations. Defaults to the dimension of A.

    Returns
    -------
    Vector
        Approximate solution x such that ||b - Ax|| is within tolerance.
    """
    A, b = system.A, system.b
    n = len(A)

    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = [0.0 for _ in range(n)]
    else:
        if len(x0) != n:
            raise ValueError("Initial guess x0 has incorrect dimension.")
        x = list(x0)  # make a mutable copy

    r = _axpy(-1.0, _matvec(A, x), b)  # r = b - A*x
    p = r.copy()
    rs_old = _dot(r, r)

    if rs_old < tol * tol:
        return x  # initial guess already good

    for iteration in range(max_iter):
        Ap = _matvec(A, p)
        pAp = _dot(p, Ap)
        if pAp == 0.0:
            raise RuntimeError("Breakdown: p^T A p == 0")
        alpha = rs_old / pAp
        x = _axpy(alpha, p, x)          # x_{k+1} = x_k + alpha * p_k
        r = _axpy(-alpha, Ap, r)        # r_{k+1} = r_k - alpha * A p_k
        rs_new = _dot(r, r)

        # Check convergence using relative residual norm
        if rs_new ** 0.5 <= tol * _norm(b):
            break

        beta = rs_new / rs_old
        p = _axpy(beta, p, r)           # p_{k+1} = r_{k+1} + beta * p_k
        rs_old = rs_new

    return x
