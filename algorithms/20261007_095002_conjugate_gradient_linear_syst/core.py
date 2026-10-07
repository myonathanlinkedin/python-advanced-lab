"""
Conjugate Gradient Linear System Solver.

Implements the Conjugate Gradient (CG) method for solving symmetric positive-definite
linear systems Ax = b using only the Python standard library.
"""

from typing import List, Tuple, Optional
import math


def dot(a: List[float], b: List[float]) -> float:
    """Compute the dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def axpy(alpha: float, x: List[float], y: List[float]) -> List[float]:
    """Compute y + alpha * x (vector addition with scaling)."""
    return [yi + alpha * xi for xi, yi in zip(x, y)]


def scale(alpha: float, x: List[float]) -> List[float]:
    """Scale a vector by a scalar."""
    return [alpha * xi for xi in x]


def vec_sub(a: List[float], b: List[float]) -> List[float]:
    """Compute a - b element-wise."""
    return [ai - bi for ai, bi in zip(a, b)]


def vec_add(a: List[float], b: List[float]) -> List[float]:
    """Compute a + b element-wise."""
    return [ai + bi for ai, bi in zip(a, b)]


def mat_vec(A: List[List[float]], v: List[float]) -> List[float]:
    """Multiply matrix A by vector v."""
    n = len(v)
    return [sum(A[i][j] * v[j] for j in range(n)) for i in range(len(A))]


def norm(x: List[float]) -> float:
    """Compute the Euclidean (L2) norm of a vector."""
    return math.sqrt(sum(xi * xi for xi in x))


def residual(A: List[List[float]], x: List[float], b: List[float]) -> List[float]:
    """Compute the residual r = b - Ax."""
    Ax = mat_vec(A, x)
    return vec_sub(b, Ax)


def conjugate_gradient(
    A: List[List[float]],
    b: List[float],
    x0: Optional[List[float]] = None,
    tol: float = 1e-10,
    max_iter: int = 1000
) -> Tuple[List[float], int, float]:
    """
    Solve Ax = b using the Conjugate Gradient method.

    Args:
        A: Symmetric positive-definite matrix (n x n).
        b: Right-hand side vector.
        x0: Initial guess (defaults to zero vector).
        tol: Convergence tolerance on the residual norm.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of (solution, iterations_used, final_residual_norm).
    """
    n = len(b)
    if x0 is None:
        x = [0.0] * n
    else:
        x = list(x0)

    r = residual(A, x, b)
    p = list(r)
    rs_old = dot(r, r)
    b_norm = norm(b)

    # If b is zero, return zero solution immediately
    if b_norm < tol:
        return x, 0, 0.0

    for k in range(1, max_iter + 1):
        Ap = mat_vec(A, p)
        pAp = dot(p, Ap)

        if pAp <= 0:
            # Matrix is not positive definite; break to avoid division by zero
            break

        alpha = rs_old / pAp
        x = axpy(alpha, p, x)
        r = axpy(-alpha, Ap, r)

        r_norm = norm(r)
        if r_norm < tol:
            return x, k, r_norm

        rs_new = dot(r, r)
        beta = rs_new / rs_old
        p = axpy(beta, p, r)
        rs_old = rs_new

    return x, max_iter, norm(r)


def is_symmetric(A: List[List[float]], tol: float = 1e-12) -> bool:
    """Check if a matrix is symmetric within tolerance."""
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False
    return True


def is_positive_definite(A: List[List[float]], tol: float = 1e-12) -> bool:
    """
    Check if a matrix is positive definite using Cholesky decomposition.
    Returns True if the matrix is SPD, False otherwise.
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= tol:
                    return False
                L[i][j] = math.sqrt(val)
            else:
                if L[j][j] == 0:
                    return False
                L[i][j] = (A[i][j] - s) / L[j][j]
    return True
