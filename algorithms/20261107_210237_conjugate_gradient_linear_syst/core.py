import math
from typing import List, Optional

def dot(u: List[float], v: List[float]) -> float:
    if len(u) != len(v):
        raise ValueError("Vectors must be same length")
    return sum(ui * vi for ui, vi in zip(u, v))

def matvec(A: List[List[float]], x: List[float]) -> List[float]:
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("Matrix must be square")
    if len(x) != n:
        raise ValueError("Vector length must match matrix dimension")
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]

def conjugate_gradient(A: List[List[float]], b: List[float], x0: Optional[List[float]] = None,
                       tol: float = 1e-10, max_iter: Optional[int] = None) -> List[float]:
    n = len(A)
    if len(b) != n:
        raise ValueError("b length must match A dimension")
    if x0 is None:
        x = [0.0] * n
    else:
        if len(x0) != n:
            raise ValueError("x0 length must match A dimension")
        x = x0.copy()
    r = [b_i - ai for b_i, ai in zip(b, matvec(A, x))]
    p = r.copy()
    rsold = dot(r, r)
    if max_iter is None:
        max_iter = n
    for _ in range(max_iter):
        Ap = matvec(A, p)
        alpha = rsold / dot(p, Ap)
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * api for ri, api in zip(r, Ap)]
        rsnew = dot(r, r)
        if math.sqrt(rsnew) < tol:
            break
        p = [ri + (rsnew / rsold) * pi for ri, pi in zip(r, p)]
        rsold = rsnew
    return x
