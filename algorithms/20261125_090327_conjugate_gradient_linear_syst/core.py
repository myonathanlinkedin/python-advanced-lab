"""
Conjugate Gradient (CG) solver for symmetric positive‑definite (SPD) linear systems.

The implementation works with dense matrices represented as ``list[list[float]]``
or with a user‑provided matrix‑vector product callable.  No external libraries are
required; all linear‑algebra primitives are implemented explicitly.

Public API
----------
- ``conjugate_gradient`` – solve ``A x = b`` using CG.
- ``dot`` – Euclidean inner product of two vectors.
- ``matvec`` – matrix‑vector product for dense matrices.
"""

from __future__ import annotations
from typing import Callable, List, Sequence, Tuple, Union

Vector = List[float]
Matrix = List[List[float]]
MatVec = Callable[[Vector], Vector]

def dot(u: Sequence[float], v: Sequence[float]) -> float:
    """Return the Euclidean inner product of two vectors."""
    if len(u) != len(v):
        raise ValueError("Vectors must be the same length for dot product.")
    return sum(au * av for au, av in zip(u, v))

def matvec(A: Matrix, x: Vector) -> Vector:
    """Dense matrix‑vector multiplication."""
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("Matrix must be square.")
    if len(x) != n:
        raise ValueError("Dimension mismatch between matrix and vector.")
    return [dot(row, x) for row in A]

def _as_matvec(A: Union[Matrix, MatVec]) -> MatVec:
    """Wrap a dense matrix as a callable matvec, otherwise return the callable."""
    if callable(A):
        return A  # type: ignore[arg-type]
    return lambda x: matvec(A, x)  # type: ignore[return-value]

def conjugate_gradient(
    A: Union[Matrix, MatVec],
    b: Vector,
    x0: Vector | None = None,
    *,
    tol: float = 1e-8,
    max_iter: int | None = None,
) -> Tuple[Vector, int]:
    """
    Solve the linear system ``A x = b`` for *x* using the Conjugate Gradient method.

    Parameters
    ----------
    A
        Either a square dense matrix (list of lists) or a callable that returns
        ``A @ v`` for a given vector *v*.
    b
        Right‑hand side vector.
    x0
        Optional initial guess.  If omitted, a zero vector is used.
    tol
        Desired relative residual norm reduction; iteration stops when
        ``||r_k|| / ||b|| <= tol``.
    max_iter
        Upper bound on iterations.  Defaults to ``len(b)`` which is sufficient
        for exact arithmetic on an *n*‑dimensional SPD system.

    Returns
    -------
    x
        Approximate solution vector.
    iters
        Number of iterations performed.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent or ``tol`` is non‑positive.
    """
    if tol <= 0.0:
        raise ValueError("Tolerance must be positive.")
    n = len(b)
    if n == 0:
        return [], 0
    if any(len(row) != n for row in (A if isinstance(A, list) else [])):
        raise ValueError("Matrix A must be square and match size of b.")
    matvec_fn = _as_matvec(A)

    x = [0.0] * n if x0 is None else list(x0)
    if len(x) != n:
        raise ValueError("Initial guess x0 must have same length as b.")
    r = [bi - ai for bi, ai in zip(b, matvec_fn(x))]
    p = r.copy()
    rs_old = dot(r, r)
    b_norm = dot(b, b) ** 0.5
    if b_norm == 0.0:
        # Trivial system 0 = 0
        return x, 0
    max_iter = n if max_iter is None else max_iter

    for it in range(1, max_iter + 1):
        Ap = matvec_fn(p)
        pAp = dot(p, Ap)
        if pAp <= 0.0:
            raise ValueError("Matrix is not symmetric positive‑definite.")
        alpha = rs_old / pAp
        # x_{k+1} = x_k + alpha * p_k
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        # r_{k+1} = r_k - alpha * A p_k
        r = [ri - alpha * api for ri, api in zip(r, Ap)]
        rs_new = dot(r, r)
        if (rs_new ** 0.5) / b_norm <= tol:
            return x, it
        beta = rs_new / rs_old
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rs_old = rs_new
    return x, max_iter

__all__ = ["conjugate_gradient", "dot", "matvec"]
