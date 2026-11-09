from __future__ import annotations
from typing import List
import math
from custom_types import Vector, Matrix


def dot(u: Vector, v: Vector) -> float:
    """Compute dot product of two vectors."""
    return sum(ux * vx for ux, vx in zip(u, v))


def mat_vec_mul(A: Matrix, x: Vector) -> Vector:
    """Multiply matrix A by vector x."""
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in A]


def vector_add(u: Vector, v: Vector) -> Vector:
    """Add two vectors."""
    return [ux + vx for ux, vx in zip(u, v)]


def vector_sub(u: Vector, v: Vector) -> Vector:
    """Subtract vector v from u."""
    return [ux - vx for ux, vx in zip(u, v)]


def scalar_mul(s: float, v: Vector) -> Vector:
    """Multiply vector v by scalar s."""
    return [s * vi for vi in v]


def norm(v: Vector) -> float:
    """Euclidean norm of vector v."""
    return math.sqrt(dot(v, v))


class ConjugateGradientSolver:
    """Conjugate Gradient solver for symmetric positive definite linear systems."""

    def __init__(self, A: Matrix, tol: float = 1e-10, max_iter: int | None = None) -> None:
        self.A = A
        self.tol = tol
        self.max_iter = max_iter or len(A)
        self._validate_matrix()

    def _validate_matrix(self) -> None:
        n = len(self.A)
        if any(len(row) != n for row in self.A):
            raise ValueError("Matrix A must be square.")
        # Check symmetry
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.A[i][j] - self.A[j][i]) > 1e-12:
                    raise ValueError("Matrix A must be symmetric.")
        # Check positive definiteness via leading principal minors (simple test)
        for k in range(1, n + 1):
            det = self._determinant(self.A[:k][:k])
            if det <= 0:
                raise ValueError("Matrix A must be positive definite.")

    def _determinant(self, M: Matrix) -> float:
        """Compute determinant of a small matrix via Gaussian elimination."""
        n = len(M)
        mat = [row[:] for row in M]
        det = 1.0
        for i in range(n):
            pivot = None
            for j in range(i, n):
                if abs(mat[j][i]) > 1e-12:
                    pivot = j
                    break
            if pivot is None:
                return 0.0
            if pivot != i:
                mat[i], mat[pivot] = mat[pivot], mat[i]
                det *= -1
            det *= mat[i][i]
            pivot_val = mat[i][i]
            for j in range(i + 1, n):
                factor = mat[j][i] / pivot_val
                for k in range(i, n):
                    mat[j][k] -= factor * mat[i][k]
        return det

    def solve(self, b: Vector, x0: Vector | None = None) -> Vector:
        """Solve Ax = b using Conjugate Gradient."""
        n = len(self.A)
        if len(b) != n:
            raise ValueError("Dimension mismatch between A and b.")
        if x0 is None:
            x = [0.0] * n
        else:
            if len(x0) != n:
                raise ValueError("Dimension mismatch between A and initial guess.")
            x = x0[:]
        r = vector_sub(b, mat_vec_mul(self.A, x))
        p = r[:]
        rsold = dot(r, r)
        if math.sqrt(rsold) < self.tol:
            return x
        for _ in range(self.max_iter):
            Ap = mat_vec_mul(self.A, p)
            alpha = rsold / dot(p, Ap)
            x = vector_add(x, scalar_mul(alpha, p))
            r = vector_sub(r, scalar_mul(alpha, Ap))
            rsnew = dot(r, r)
            if math.sqrt(rsnew) < self.tol:
                break
            p = vector_add(r, scalar_mul(rsnew / rsold, p))
            rsold = rsnew
        return x
