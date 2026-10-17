"""
Conjugate Gradient (CG) solver for symmetric positive‑definite (SPD) linear systems.

The module provides:
- Basic linear‑algebra utilities (dot product, matrix‑vector product, norm).
- A simple Cholesky‑based SPD validator.
- `conjugate_gradient` function implementing the CG algorithm.

All data structures are plain Python lists; no external dependencies are used.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional


Vector = List[float]
Matrix = List[List[float]]


def dot(u: Vector, v: Vector) -> float:
    """Return the Euclidean dot product of two vectors."""
    return sum(au * av for au, av in zip(u, v))


def matvec(A: Matrix, x: Vector) -> Vector:
    """Return the matrix‑vector product A·x."""
    return [dot(row, x) for row in A]


def norm(v: Vector) -> float:
    """Return the Euclidean norm of a vector."""
    return math.sqrt(dot(v, v))


def _cholesky(A: Matrix) -> Optional[Matrix]:
    """
    Attempt a Cholesky decomposition of a symmetric matrix A.
    Returns lower‑triangular L such that A = L·Lᵀ if successful,
    otherwise returns None (indicating non‑SPD).
    """
    n = len(A)
    L: Matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = A[i][j]
            for k in range(j):
                s -= L[i][k] * L[j][k]

            if i == j:
                if s <= 0.0:
                    return None
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def is_spd(A: Matrix, *, tol: float = 1e-12) -> bool:
    """
    Verify that a matrix is symmetric positive‑definite.
    Symmetry is checked within `tol`; positive‑definiteness via Cholesky.
    """
    n = len(A)
    if any(len(row) != n for row in A):
        return False

    # Symmetry check
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False

    # Positive‑definiteness via Cholesky
    return _cholesky(A) is not None


def conjugate_gradient(
    A: Matrix,
    b: Vector,
    x0: Optional[Vector] = None,
    *,
    tol: float = 1e-8,
    max_iter: Optional[int] = None,
) -> Tuple[Vector, int, float]:
    """
    Solve the linear system A·x = b using the Conjugate Gradient method.

    Parameters
    ----------
    A : Matrix
        Symmetric positive‑definite coefficient matrix.
    b : Vector
        Right‑hand side vector.
    x0 : Vector, optional
        Initial guess (defaults to the zero vector).
    tol : float, optional
        Desired residual norm tolerance.
    max_iter : int, optional
        Maximum number of iterations (defaults to the dimension of A).

    Returns
    -------
    x : Vector
        Approximate solution.
    iters : int
        Number of iterations performed.
    final_residual : float
        Euclidean norm of the final residual.
    """
    if not is_spd(A):
        raise ValueError("Matrix A must be symmetric positive‑definite.")

    n = len(A)
    if len(b) != n:
        raise ValueError("Dimension mismatch between A and b.")

    if x0 is None:
        x = [0.0] * n
    else:
        if len(x0) != n:
            raise ValueError("Initial guess x0 has incorrect dimension.")
        x = x0[:]

    r = [bi - ai for bi, ai in zip(b, matvec(A, x))]
    p = r[:]
    rsold = dot(r, r)

    if math.sqrt(rsold) < tol:
        return x, 0, math.sqrt(rsold)

    if max_iter is None:
        max_iter = n

    for it in range(1, max_iter + 1):
        Ap = matvec(A, p)
        pAp = dot(p, Ap)
        if pAp == 0.0:
            raise RuntimeError("Breakdown: pᵀ·A·p == 0")
        alpha = rsold / pAp

        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * Api for ri, Api in zip(r, Ap)]

        rsnew = dot(r, r)
        residual = math.sqrt(rsnew)
        if residual < tol:
            return x, it, residual

        beta = rsnew / rsold
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rsold = rsnew

    return x, max_iter, math.sqrt(rsold)
