from __future__ import annotations
from typing import Tuple
import math
from .types import Matrix, Vector, SVDResult

def transpose(m: Matrix) -> Matrix:
    if not m:
        return []
    return [list(col) for col in zip(*m)]

def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    """Multiply two matrices a (m x p) and b (p x n)."""
    assert a and b and len(a[0]) == len(b), "Incompatible dimensions for multiplication"
    result: Matrix = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
    b_t = transpose(b)
    for i, row in enumerate(a):
        for j, col in enumerate(b_t):
            result[i][j] = sum(x * y for x, y in zip(row, col))
    return result

def mat_vec_mul(m: Matrix, v: Vector) -> Vector:
    """Multiply matrix m (m x n) by vector v (n)."""
    assert m and len(m[0]) == len(v), "Incompatible dimensions for mat-vec multiplication"
    return [sum(row[i] * v[i] for i in range(len(v))) for row in m]

def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def frobenius_norm(m: Matrix) -> float:
    return math.sqrt(sum(x * x for row in m for x in row))

def max_offdiag(s: Matrix) -> Tuple[int, int, float]:
    """Return indices (p, q) of the largest off‑diagonal absolute element and its value."""
    n = len(s)
    max_val = 0.0
    p = q = -1
    for i in range(n):
        for j in range(i + 1, n):
            val = abs(s[i][j])
            if val > max_val:
                max_val = val
                p, q = i, j
    return p, q, max_val

def jacobi_eigen(s: Matrix, eps: float = 1e-12, max_iter: int = 100) -> Tuple[Vector, Matrix]:
    """
    Jacobi eigenvalue algorithm for a real symmetric matrix.
    Returns eigenvalues (list) and eigenvectors (matrix whose columns are eigenvectors).
    """
    n = len(s)
    a = [row[:] for row in s]          # working copy
    v = identity(n)                    # eigenvector matrix
    for _ in range(max_iter):
        p, q, max_val = max_offdiag(a)
        if max_val < eps:
            break
        phi = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        c = math.cos(phi)
        s_ = math.sin(phi)

        # Rotate rows and columns p and q in matrix a
        for i in range(n):
            if i != p and i != q:
                api = a[p][i]
                aqi = a[q][i]
                a[p][i] = a[i][p] = c * api - s_ * aqi
                a[q][i] = a[i][q] = s_ * api + c * aqi

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]

        a[p][p] = c * c * app - 2.0 * s_ * c * apq + s_ * s_ * aqq
        a[q][q] = s_ * s_ * app + 2.0 * s_ * c * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0

        # Update eigenvectors
        for i in range(n):
            vip = v[i][p]
            viq = v[i][q]
            v[i][p] = c * vip - s_ * viq
            v[i][q] = s_ * vip + c * viq

    eigenvalues = [a[i][i] for i in range(n)]
    # Normalize eigenvectors (columns of v)
    for j in range(n):
        norm = math.sqrt(sum(v[i][j] * v[i][j] for i in range(n)))
        if norm > 0:
            for i in range(n):
                v[i][j] /= norm
    return eigenvalues, v

def svd(A: Matrix, eps: float = 1e-12) -> SVDResult:
    """
    Compute the singular value decomposition of A (m x n) using
    eigen‑decomposition of AᵀA (the classic textbook method).
    Returns U (m x r), singular values (length r), and Vᵀ (r x n).
    """
    m, n = len(A), len(A[0]) if A else 0
    At = transpose(A)
    AtA = mat_mul(At, A)                     # n x n symmetric
    eigenvals, V = jacobi_eigen(AtA, eps=eps)

    # Singular values are sqrt of non‑negative eigenvalues
    singulars = [math.sqrt(max(val, 0.0)) for val in eigenvals]

    # Sort descending by singular value, keep the same permutation for V
    idx = sorted(range(len(singulars)), key=lambda i: singulars[i], reverse=True)
    singulars = [singulars[i] for i in idx]
    V = [[V[row][i] for i in idx] for row in range(n)]

    # Build U = A * V * Σ⁻¹
    rank = sum(1 for s in singulars if s > eps)
    U: Matrix = [[0.0 for _ in range(rank)] for _ in range(m)]
    for i in range(rank):
        sigma = singulars[i]
        if sigma < eps:
            continue
        v_i = [V[row][i] for row in range(n)]          # column i of V
        Av = mat_vec_mul(A, v_i)                       # m‑vector
        for row in range(m):
            U[row][i] = Av[row] / sigma

    # Vᵀ is the transpose of V (only the first `rank` columns are needed)
    Vt = [ [V[col][row] for col in range(rank)] for row in range(n) ]
    return SVDResult(u=U, s=singulars, vt=Vt)

def low_rank_approx(A: Matrix, k: int) -> Matrix:
    """
    Return the rank‑k approximation of A using its singular value decomposition.
    """
    assert k >= 0, "k must be non‑negative"
    svd_res = svd(A)
    m, n = len(A), len(A[0]) if A else 0
    k = min(k, len(svd_res.s))
    # Build Σ_k
    sigma_k = [[0.0 for _ in range(k)] for _ in range(k)]
    for i in range(k):
        sigma_k[i][i] = svd_res.s[i]

    # Truncate U and Vᵀ
    U_k = [row[:k] for row in svd_res.u]          # m x k
    Vt_k = [row[:k] for row in svd_res.vt]        # k x n

    # Compute U_k * Σ_k
    US = mat_mul(U_k, sigma_k)                    # m x k
    # Compute (U_k * Σ_k) * Vᵀ_k
    approx = mat_mul(US, Vt_k)                    # m x n
    return approx
