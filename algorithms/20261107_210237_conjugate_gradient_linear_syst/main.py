import math
import random
from typing import List
from core import conjugate_gradient, matvec, dot

def is_symmetric(A: List[List[float]]) -> bool:
    n = len(A)
    return all(A[i][j] == A[j][i] for i in range(n) for j in range(i+1, n))

def generate_spd_matrix(n: int, seed: int = 0) -> List[List[float]]:
    random.seed(seed)
    M = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    A = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = sum(M[k][i]*M[k][j] for k in range(n))
    for i in range(n):
        A[i][i] += n
    return A

def test_small_spd():
    A = [[4.0, 1.0], [1.0, 3.0]]
    b = [1.0, 2.0]
    x = conjugate_gradient(A, b, tol=1e-12, max_iter=10)
    expected = [1.0/11.0, 7.0/11.0]
    assert all(abs(xi - ei) < 1e-10 for xi, ei in zip(x, expected)), f"Expected {expected}, got {x}"

def test_random_spd():
    n = 10
    A = generate_spd_matrix(n, seed=42)
    b = [random.uniform(-5,5) for _ in range(n)]
    x = conjugate_gradient(A, b, tol=1e-8, max_iter=2*n)
    r = [bi - sum(A[i][j]*x[j] for j in range(n)) for i, bi in enumerate(b)]
    res_norm = math.sqrt(dot(r, r))
    assert res_norm < 1e-6, f"Residual norm too high: {res_norm}"

def test_dimension_mismatch():
    A = [[1.0]]
    b = [1.0, 2.0]
    try:
        conjugate_gradient(A, b)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for dimension mismatch"

def test_initial_guess():
    A = [[2.0, 0.0], [0.0, 2.0]]
    b = [2.0, 2.0]
    x0 = [1.0, 1.0]
    x = conjugate_gradient(A, b, x0=x0, tol=1e-12, max_iter=5)
    expected = [1.0, 1.0]
    assert all(abs(xi - ei) < 1e-12 for xi, ei in zip(x, expected)), f"Expected {expected}, got {x}"

def run_tests():
    test_small_spd()
    test_random_spd()
    test_dimension_mismatch()
    test_initial_guess()
    print("All tests passed.")

if __name__ == "__main__":
    run_tests()
