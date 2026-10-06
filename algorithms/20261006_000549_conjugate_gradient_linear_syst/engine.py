from __future__ import annotations
from typing import List, Optional
from custom_types import Vector, Matrix, dot, norm, scalar_mul, vec_add, vec_sub, mat_vec_mul, is_square_matrix


class ConjugateGradientSolver:
    """
    Conjugate Gradient solver for symmetric positive definite matrices.
    """

    def __init__(self, A: Matrix, tol: float = 1e-10, max_iter: Optional[int] = None) -> None:
        if not is_square_matrix(A):
            raise ValueError("Matrix A must be square.")
        self.A = A
        self.n = len(A)
        self.tol = tol
        self.max_iter = max_iter or self.n

    def solve(self, b: Vector, x0: Optional[Vector] = None) -> Vector:
        if len(b) != self.n:
            raise ValueError("Dimension mismatch between A and b.")
        if x0 is None:
            x = [0.0] * self.n
        else:
            if len(x0) != self.n:
                raise ValueError("Initial guess x0 must have same dimension as b.")
            x = x0.copy()

        r = vec_sub(b, mat_vec_mul(self.A, x))
        p = r.copy()
        rs_old = dot(r, r)

        if math.sqrt(rs_old) < self.tol:
            return x

        for iteration in range(self.max_iter):
            Ap = mat_vec_mul(self.A, p)
            alpha = rs_old / dot(p, Ap)
            x = vec_add(x, scalar_mul(alpha, p))
            r = vec_sub(r, scalar_mul(alpha, Ap))
            rs_new = dot(r, r)
            if math.sqrt(rs_new) < self.tol:
                return x
            beta = rs_new / rs_old
            p = vec_add(r, scalar_mul(beta, p))
            rs_old = rs_new

        raise RuntimeError("Conjugate Gradient did not converge within the maximum number of iterations.")
