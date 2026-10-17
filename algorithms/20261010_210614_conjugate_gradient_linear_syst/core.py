from __future__ import annotations
import math
from typing import List, Sequence, Optional

Vector = List[float]
Matrix = List[List[float]]


def _dot(u: Sequence[float], v: Sequence[float]) -> float:
    """Return the Euclidean dot product of two vectors."""
    return sum(ui * vi for ui, vi in zip(u, v))


def _norm(v: Sequence[float]) -> float:
    """Return the Euclidean norm of a vector."""
    return math.sqrt(_dot(v, v))


def _add(u: Sequence[float], v: Sequence[float]) -> Vector:
    """Element‑wise addition of two vectors."""
    return [ui + vi for ui, vi in zip(u, v)]


def _sub(u: Sequence[float], v: Sequence[float]) -> Vector:
    """Element‑wise subtraction of two vectors (u - v)."""
    return [ui - vi for ui, vi in zip(u, v)]


def _scalar_mul(alpha: float, v: Sequence[float]) -> Vector:
    """Multiply a vector by a scalar."""
    return [alpha * vi for vi in v]


def _matvec(A: Matrix, x: Sequence[float]) -> Vector:
    """Matrix‑vector multiplication y = A·x."""
    return [_dot(row, x) for row in A]


class ConjugateGradientSolver:
    """
    Solver for symmetric positive‑definite linear systems using the
    Conjugate Gradient method.
    """

    @staticmethod
    def solve(
        A: Matrix,
        b: Vector,
        x0: Optional[Vector] = None,
        tol: float = 1e-8,
        max_iter: Optional[int] = None,
    ) -> Vector:
        """
        Solve Ax = b.

        Parameters
        ----------
        A : Matrix
            Symmetric positive‑definite matrix (n×n).
        b : Vector
            Right‑hand side vector (length n).
        x0 : Vector, optional
            Initial guess (defaults to zero vector).
        tol : float, optional
            Desired residual norm tolerance.
        max_iter : int, optional
            Maximum number of iterations (defaults to n).

        Returns
        -------
        x : Vector
            Approximate solution.
        """
        n = len(A)
        if any(len(row) != n for row in A):
            raise ValueError("Matrix A must be square.")
        if len(b) != n:
            raise ValueError("Dimension mismatch between A and b.")

        if max_iter is None:
            max_iter = n * 10

        x = [0.0] * n if x0 is None else list(x0)
        r = _sub(b, _matvec(A, x))
        p = r.copy()
        rsold = _dot(r, r)

        for _ in range(max_iter):
            Ap = _matvec(A, p)
            pAp = _dot(p, Ap)
            if pAp == 0.0:
                # Should not happen for SPD matrices unless convergence reached.
                break
            alpha = rsold / pAp
            x = _add(x, _scalar_mul(alpha, p))
            r = _sub(r, _scalar_mul(alpha, Ap))
            rsnew = _dot(r, r)
            if math.sqrt(rsnew) < tol:
                break
            beta = rsnew / rsold
            p = _add(r, _scalar_mul(beta, p))
            rsold = rsnew

        return x
