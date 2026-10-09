"""
Conjugate Gradient (CG) solver for symmetric positive-definite (SPD) linear systems.

Provides:
- `ConjugateGradient` class with a deterministic, pure‑Python implementation.
- Helper functions for dense matrix‑vector multiplication, dot product, and Euclidean norm.
- Type‑hints and exhaustive docstrings for clarity and verification.
"""

from __future__ import annotations
from typing import Callable, List, Sequence, Union, Optional

Vector = List[float]
Matrix = List[List[float]]
MatVec = Callable[[Vector], Vector]


def _dot(u: Sequence[float], v: Sequence[float]) -> float:
    """Return the Euclidean inner product of two vectors."""
    return sum(ua * va for ua, va in zip(u, v))


def _norm(v: Sequence[float]) -> float:
    """Return the Euclidean (L2) norm of a vector."""
    return _dot(v, v) ** 0.5


def _matvec_dense(A: Matrix, x: Vector) -> Vector:
    """Matrix‑vector product for a dense matrix represented as a list of lists."""
    return [_dot(row, x) for row in A]


def _is_callable(A: Union[Matrix, MatVec]) -> bool:
    """Return True if A is a callable (i.e., a matrix‑vector product function)."""
    return callable(A)


class ConjugateGradient:
    """
    Solver for Ax = b where A is symmetric positive‑definite.

    Parameters
    ----------
    A : Union[Matrix, MatVec]
        Either a dense SPD matrix (list of lists) or a callable that computes A·v.
    tol : float, optional
        Desired relative residual tolerance. Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations. If None, defaults to the dimension of A.
    """

    def __init__(
        self,
        A: Union[Matrix, MatVec],
        *,
        tol: float = 1e-8,
        max_iter: Optional[int] = None,
    ) -> None:
        self.A = A
        self.tol = float(tol)
        self.max_iter = max_iter

    def _apply_A(self, x: Vector) -> Vector:
        """Apply the linear operator A to vector x."""
        if _is_callable(self.A):
            return self.A(x)  # type: ignore[arg-type]
        else:
            return _matvec_dense(self.A, x)

    def solve(
        self,
        b: Vector,
        x0: Optional[Vector] = None,
    ) -> Vector:
        """
        Solve Ax = b using the Conjugate Gradient method.

        Parameters
        ----------
        b : Vector
            Right‑hand side vector.
        x0 : Vector, optional
            Initial guess. If None, uses the zero vector.

        Returns
        -------
        x : Vector
            Approximate solution satisfying the tolerance criteria.
        """
        n = len(b)
        if x0 is None:
            x = [0.0] * n
        else:
            if len(x0) != n:
                raise ValueError("Initial guess x0 must have same dimension as b.")
            x = list(x0)

        r = [bi - ai for bi, ai in zip(b, self._apply_A(x))]
        p = r.copy()
        rs_old = _dot(r, r)

        if rs_old == 0.0:
            return x  # b is zero vector; solution is trivial.

        # Determine iteration limit.
        max_iter = self.max_iter if self.max_iter is not None else n

        for iteration in range(max_iter):
            Ap = self._apply_A(p)
            pAp = _dot(p, Ap)

            if pAp <= 0.0:
                raise ValueError(
                    f"Matrix is not positive‑definite (p·Ap={pAp}) at iteration {iteration}."
                )

            alpha = rs_old / pAp

            # x_{k+1} = x_k + α p_k
            x = [xi + alpha * pi for xi, pi in zip(x, p)]

            # r_{k+1} = r_k - α A p_k
            r = [ri - alpha * api for ri, api in zip(r, Ap)]
            rs_new = _dot(r, r)

            # Check convergence: relative residual norm.
            if _norm(r) <= self.tol * _norm(b):
                break

            beta = rs_new / rs_old
            p = [ri + beta * pi for ri, pi in zip(r, p)]
            rs_old = rs_new

        return x
