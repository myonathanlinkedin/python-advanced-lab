"""Conjugate Gradient (CG) solver for symmetric positive-definite linear systems.

Implements the classic Fletcher-Reeves CG iteration with residual-based
convergence testing.  All operations use only the Python standard library.

Time complexity:  O(n^2) per iteration (matrix-vector product), O(n^2 * k) total
Space complexity: O(n) for vectors, O(n^2) for the matrix (dense storage).
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional


# Type alias: a dense matrix is a list of rows, each row a list of floats.
Matrix = List[List[float]]
Vector = List[float]


def matvec(A: Matrix, x: Vector) -> Vector:
    """Compute y = A @ x for a dense matrix A and vector x.

    Time: O(n^2), Space: O(n).
    """
    n = len(A)
    y = [0.0] * n
    for i in range(n):
        row = A[i]
        s = 0.0
        for j in range(n):
            s += row[j] * x[j]
        y[i] = s
    return y


def dot(u: Vector, v: Vector) -> float:
    """Euclidean inner product u · v.  Time: O(n)."""
    s = 0.0
    for i in range(len(u)):
        s += u[i] * v[i]
    return s


def axpy(alpha: float, x: Vector, y: Vector) -> Vector:
    """Return y + alpha * x (new vector, inputs unchanged).  Time: O(n)."""
    n = len(x)
    out = [0.0] * n
    for i in range(n):
        out[i] = y[i] + alpha * x[i]
    return out


def scale(alpha: float, x: Vector) -> Vector:
    """Return alpha * x.  Time: O(n)."""
    return [alpha * xi for xi in x]


def vec_sub(u: Vector, v: Vector) -> Vector:
    """Return u - v.  Time: O(n)."""
    return [ui - vi for ui, vi in zip(u, v)]


def vec_add(u: Vector, v: Vector) -> Vector:
    """Return u + v.  Time: O(n)."""
    return [ui + vi for ui, vi in zip(u, v)]


def norm_inf(x: Vector) -> float:
    """Infinity (max) norm of a vector.  Time: O(n)."""
    return max(abs(xi) for xi in x) if x else 0.0


def norm_2(x: Vector) -> float:
    """Euclidean (2) norm of a vector.  Time: O(n)."""
    return math.sqrt(dot(x, x))


def is_spd(A: Matrix) -> bool:
    """Check whether A is symmetric positive-definite via Cholesky factorisation.

    Returns True iff A is symmetric and all leading principal minors are
    positive (equivalently, Cholesky succeeds without encountering a
    non-positive pivot).

    Time: O(n^3), Space: O(n^2).
    """
    n = len(A)
    # Symmetry check
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] != A[j][i]:
                return False
    # Cholesky: A = L L^T
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i][k] * L[j][k]
            if i == j:
                val = A[i][i] - s
                if val <= 0.0:
                    return False
                L[i][i] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return True


def conjugate_gradient(
    A: Matrix,
    b: Vector,
    x0: Optional[Vector] = None,
    tol: float = 1e-10,
    max_iter: int = 0,
) -> Tuple[Vector, int, float]:
    """Solve A x = b using the Conjugate Gradient method.

    Parameters
    ----------
    A : (n, n) symmetric positive-definite matrix.
    b : right-hand-side vector of length n.
    x0 : initial guess (default: zero vector).
    tol : convergence tolerance on ||r||_2 / ||b||_2.
    max_iter : maximum number of iterations (default: n).

    Returns
    -------
    (x, iters, rel_residual)
        x            – approximate solution vector.
        iters        – number of iterations actually performed.
        rel_residual – final relative residual ||r||_2 / ||b||_2.

    Raises
    ------
    ValueError if A is not square, dimensions mismatch, or A is not SPD.
    """
    n = len(A)
    if len(b) != n:
        raise ValueError(f"Dimension mismatch: A is {n}x{n}, b has length {len(b)}")
    for i in range(n):
        if len(A[i]) != n:
            raise ValueError(f"Row {i} of A has length {len(A[i])}, expected {n}")

    if not is_spd(A):
        raise ValueError("Matrix A is not symmetric positive-definite.")

    if x0 is None:
        x = [0.0] * n
    else:
        if len(x0) != n:
            raise ValueError(f"x0 has length {len(x0)}, expected {n}")
        x = list(x0)

    if max_iter <= 0:
        max_iter = n

    # Initial residual
    r = vec_sub(b, matvec(A, x))
    p = list(r)
    rs_old = dot(r, r)
    b_norm = norm_2(b)
    if b_norm == 0.0:
        # b is zero; x = 0 is the exact solution.
        return x, 0, 0.0

    for k in range(1, max_iter + 1):
        Ap = matvec(A, p)
        pAp = dot(p, Ap)
        if pAp == 0.0:
            break  # numerical breakdown
        alpha = rs_old / pAp
        x = axpy(alpha, p, x)
        r = axpy(-alpha, Ap, r)
        rel_res = norm_2(r) / b_norm
        if rel_res < tol:
            return x, k, rel_res
        rs_new = dot(r, r)
        beta = rs_new / rs_old
        p = axpy(beta, r, p)
        rs_old = rs_new

    rel_res = norm_2(r) / b_norm
    return x, max_iter, rel_res


def build_spd_matrix(n: int, diag: float = 2.0, off_diag: float = 1.0) -> Matrix:
    """Construct an n×n tridiagonal SPD matrix.

    Diagonal entries = diag, off-diagonal entries = off_diag.
    SPD when diag > 2 * |off_diag| (Gershgorin / diagonal dominance).

    Time: O(n^2), Space: O(n^2).
    """
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = diag
        if i > 0:
            A[i][i - 1] = off_diag
            A[i - 1][i] = off_diag
    return A


def build_random_spd(n: int, seed: int = 42) -> Matrix:
    """Build a random SPD matrix A = M M^T + n*I using a deterministic LCG.

    Uses a simple linear congruential generator (no external deps).

    Time: O(n^3), Space: O(n^2).
    """
    # Deterministic LCG
    state = seed
    def rand() -> float:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    M = [[rand() for _ in range(n)] for _ in range(n)]
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += M[i][k] * M[j][k]
            A[i][j] = s
    # Add n*I to ensure strict positive-definiteness
    for i in range(n):
        A[i][i] += n
    return A
