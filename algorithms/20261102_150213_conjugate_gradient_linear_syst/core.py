from __future__ import annotations
from typing import Callable, List, Sequence, Union, Optional
import math

Vector = List[float]
Matrix = List[List[float]]
LinearOperator = Callable[[Vector], Vector]

def _dot(u: Sequence[float], v: Sequence[float]) -> float:
    """Return the Euclidean inner product of two vectors."""
    return sum(ua * va for ua, va in zip(u, v))

def _norm(v: Sequence[float]) -> float:
    """Return the Euclidean norm of a vector."""
    return math.sqrt(_dot(v, v))

def _matvec(A: Matrix, x: Vector) -> Vector:
    """Dense matrix‑vector product."""
    return [_dot(row, x) for row in A]

def _as_operator(A: Union[Matrix, LinearOperator]) -> LinearOperator:
    """Wrap a dense matrix as a callable linear operator."""
    if callable(A):
        return A  # type: ignore[return-value]
    else:
        return lambda x: _matvec(A, x)

def conjugate_gradient(
    A: Union[Matrix, LinearOperator],
    b: Vector,
    x0: Optional[Vector] = None,
    *,
    tol: float = 1e-8,
    max_iter: Optional[int] = None,
) -> Vector:
    """
    Solve the symmetric positive‑definite linear system A·x = b
    using the Conjugate Gradient method.

    Parameters
    ----------
    A : matrix (list of lists) or callable
        Symmetric positive‑definite linear operator.
    b : right‑hand side vector.
    x0 : optional initial guess (defaults to zero vector).
    tol : convergence tolerance on the residual norm.
    max_iter : maximum number of iterations (defaults to len(b)).

    Returns
    -------
    x : approximate solution vector.

    Raises
    ------
    ValueError
        If dimensions are inconsistent or convergence is not achieved.
    """
    n = len(b)
    if any(len(row) != n for row in (A if isinstance(A, list) else [])):
        raise ValueError("Matrix A must be square and match size of b.")
    if max_iter is None:
        max_iter = n

    op = _as_operator(A)

    x = [0.0] * n if x0 is None else list(x0)
    r = [bi - ai for bi, ai in zip(b, op(x))]
    p = r.copy()
    rs_old = _dot(r, r)

    if math.isclose(rs_old, 0.0, abs_tol=tol):
        return x  # initial guess already solves the system

    for k in range(max_iter):
        Ap = op(p)
        pAp = _dot(p, Ap)
        if pAp <= 0.0:
            raise ValueError("Matrix A is not positive‑definite.")
        alpha = rs_old / pAp
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * api for ri, api in zip(r, Ap)]
        rs_new = _dot(r, r)

        if math.sqrt(rs_new) < tol:
            return x

        beta = rs_new / rs_old
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rs_old = rs_new

    raise ValueError(f"Conjugate Gradient did not converge within {max_iter} iterations.")

__all__ = ["conjugate_gradient", "Vector", "Matrix", "LinearOperator"]
