"""
Conjugate Gradient (CG) solver for symmetric positive‑definite (SPD) linear systems.

The implementation works with:
* A dense matrix given as a list of lists (row‑major).
* A callable linear operator `A(x)` returning the matrix‑vector product.

Only the Python standard library is used.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple, Union, Optional

Vector = List[float]
Matrix = List[List[float]]
LinearOperator = Callable[[Vector], Vector]
MatrixLike = Union[Matrix, LinearOperator]


def _dot(u: Sequence[float], v: Sequence[float]) -> float:
    """Return the Euclidean dot product of two vectors."""
    return sum(ua * va for ua, va in zip(u, v))


def _norm(v: Sequence[float]) -> float:
    """Return the Euclidean norm of a vector."""
    return math.sqrt(_dot(v, v))


def _matvec(A: Matrix, x: Vector) -> Vector:
    """Matrix‑vector product for a dense row‑major matrix."""
    return [_dot(row, x) for row in A]


def _apply_A(A: MatrixLike, x: Vector) -> Vector:
    """Apply matrix or linear operator to a vector."""
    if callable(A):
        return A(x)  # type: ignore[arg-type]
    return _matvec(A, x)  # type: ignore[arg-type]


def conjugate_gradient(
    A: MatrixLike,
    b: Vector,
    x0: Optional[Vector] = None,
    tol: float = 1e-8,
    max_iter: Optional[int] = None,
) -> Vector:
    """
    Solve the linear system A·x = b using the Conjugate Gradient method.

    Parameters
    ----------
    A : MatrixLike
        SPD matrix (list of lists) or a callable returning A·x.
    b : Vector
        Right‑hand side vector.
    x0 : Vector, optional
        Initial guess (defaults to the zero vector).
    tol : float, optional
        Desired relative residual norm; iteration stops when
        ||r_k|| / ||b|| <= tol.
    max_iter : int, optional
        Maximum number of iterations (defaults to len(b)).

    Returns
    -------
    x : Vector
        Approximate solution.
    """
    n = len(b)
    if x0 is None:
        x = [0.0] * n
    else:
        if len(x0) != n:
            raise ValueError("Initial guess x0 must have the same length as b.")
        x = x0[:]

    r = [bi - ai for bi, ai in zip(b, _apply_A(A, x))]
    p = r[:]
    rs_old = _dot(r, r)
    b_norm = _norm(b)
    if b_norm == 0:
        b_norm = 1.0  # avoid division by zero for homogeneous system

    if max_iter is None:
        max_iter = n

    for iteration in range(max_iter):
        Ap = _apply_A(A, p)
        alpha = rs_old / _dot(p, Ap)
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * api for ri, api in zip(r, Ap)]
        rs_new = _dot(r, r)

        if math.sqrt(rs_new) / b_norm <= tol:
            break

        beta = rs_new / rs_old
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rs_old = rs_new

    return x


__all__: Tuple[str, ...] = ("conjugate_gradient", "Vector", "Matrix", "LinearOperator")
