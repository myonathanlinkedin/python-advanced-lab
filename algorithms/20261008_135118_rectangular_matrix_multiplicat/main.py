from __future__ import annotations

import random
import sys
from typing import List

from .engine import multiply
from .types import Matrix, MatrixDimensionError, MatrixStructureError, _validate_matrix


def _naive_multiply(A: Matrix, B: Matrix) -> Matrix:
    """Reference implementation using straightforward triple loop."""
    m, k_a = _validate_matrix(A)
    k_b, n = _validate_matrix(B)
    if k_a != k_b:
        raise MatrixDimensionError
    result: Matrix = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            total = 0
            for t in range(k_a):
                total += A[i][t] * B[t][j]
            result[i][j] = total
    return result


def _random_matrix(rows: int, cols: int, low: int = -10, high: int = 10) -> Matrix:
    return [[random.randint(low, high) for _ in range(cols)] for _ in range(rows)]


def _assert_matrices_equal(C1: Matrix, C2: Matrix, eps: float = 1e-9) -> None:
    """Assert that two matrices are element‑wise equal within tolerance."""
    m1, n1 = _validate_matrix(C1)
    m2, n2 = _validate_matrix(C2)
    assert (m1, n1) == (m2, n2), f"Shape mismatch: {m1}×{n1} vs {m2}×{n2}"
    for i in range(m1):
        for j in range(n1):
            diff = abs(C1[i][j] - C2[i][j])
            assert diff <= eps, f"Mismatch at ({i},{j}): {C1[i][j]} vs {C2[i][j]}"


def run_unit_tests() -> None:
    # Test 1: Empty matrices
    empty: Matrix = []
    assert multiply(empty, empty) == [], "Multiplying two empty matrices should yield empty."

    # Test 2: 1×1 matrices
    A = [[2]]
    B = [[3]]
    assert multiply(A, B) == [[6]], "1×1 multiplication failed."

    # Test 3: Compatible rectangular matrices
    A = [[1, 2, 3],
         [4, 5, 6]]
    B = [[7, 8],
         [9, 10],
         [11, 12]]
    expected = _naive_multiply(A, B)
    result = multiply(A, B)
    _assert_matrices_equal(result, expected)

    # Test 4: Mismatched dimensions should raise
    A = [[1, 2], [3, 4]]
    B = [[5, 6, 7]]
    try:
        multiply(A, B)
        assert False, "Expected MatrixDimensionError for mismatched dimensions."
    except MatrixDimensionError:
        pass

    # Test 5: Ragged matrix detection
    A = [[1, 2], [3]]
    B = [[4, 5], [6, 7]]
    try:
        multiply(A, B)
        assert False, "Expected MatrixStructureError for ragged matrix."
    except MatrixStructureError:
        pass

    # Test 6: Randomized stress test
    for _ in range(20):
        m = random.randint(1, 5)
        k = random.randint(1, 5)
        n = random.randint(1, 5)
        A = _random_matrix(m, k)
        B = _random_matrix(k, n)
        expected = _naive_multiply(A, B)
        result = multiply(A, B)
        _assert_matrices_equal(result, expected)

    print("All unit tests passed.")


def demo() -> None:
    """Simple demonstration with a fixed example."""
    A = [[1, 4, 7],
         [2, 5, 8],
         [3, 6, 9]]
    B = [[9, 8, 7],
         [6, 5, 4],
         [3, 2, 1]]
    print("Matrix A:")
    for row in A:
        print(row)
    print("\nMatrix B:")
    for row in B:
        print(row)
    C = multiply(A, B)
    print("\nA × B =")
    for row in C:
        print(row)


if __name__ == "__main__":
    # Run verification suite first; abort on failure.
    run_unit_tests()
    # Demonstration of correct operation.
    demo()
