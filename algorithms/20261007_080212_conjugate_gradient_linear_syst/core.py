from __future__ import annotations
from typing import List, Optional

def dot(a: List[float], b: List[float]) -> float:
    """Compute dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def norm(a: List[float]) -> float:
    """Euclidean norm of a vector."""
    return dot(a, a) ** 0.5

def matvec(A: List[List[float]], x: List[float]) -> List[float]:
    """Matrix-vector multiplication for a dense matrix."""
    return [dot(row, x) for row in A]

def cg(
    A: List[List[float]],
    b: List[float],
    x0: Optional[List[float]] = None,
    tol: float = 1e-10,
    max_iter: Optional[int] = None,
) -> List[float]:
    """
    Conjugate Gradient solver for symmetric positive definite matrices.

    Parameters
    ----------
    A : List[List[float]]
        Symmetric positive definite matrix.
    b : List[float]
        Right-hand side vector.
    x0 : Optional[List[float]]
        Initial guess. If None, zeros are used.
    tol : float
        Tolerance for stopping criterion based on residual norm.
    max_iter : Optional[int]
        Maximum number of iterations. Defaults to len(b).

    Returns
    -------
    List[float]
        Approximate solution vector.
    """
    n = len(b)
    if x0 is None:
        x = [0.0] * n
    else:
        x = x0[:]
    r = [b_i - sum(A[i][j] * x[j] for j in range(n)) for i, b_i in enumerate(b)]
    p = r[:]
    rsold = dot(r, r)
    if max_iter is None:
        max_iter = n
    for _ in range(max_iter):
        Ap = matvec(A, p)
        alpha = rsold / dot(p, Ap)
        x = [x_i + alpha * p_i for x_i, p_i in zip(x, p)]
        r = [r_i - alpha * Ap_i for r_i, Ap_i in zip(r, Ap)]
        rsnew = dot(r, r)
        if rsnew ** 0.5 < tol:
            break
        p = [r_i + (rsnew / rsold) * p_i for r_i, p_i in zip(r, p)]
        rsold = rsnew
    return x
