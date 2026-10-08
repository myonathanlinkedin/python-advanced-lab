from __future__ import annotations
from typing import List
from .types import CSRMatrix, Vector
from .engine import spmv


def dense_spmv_dense(matrix: List[List[float]], vector: Vector) -> Vector:
    """Reference dense implementation for verification."""
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows else 0
    if len(vector) != n_cols:
        raise ValueError("Dimension mismatch in dense reference")
    result = [0.0 for _ in range(n_rows)]
    for i in range(n_rows):
        s = 0.0
        for j in range(n_cols):
            s += matrix[i][j] * vector[j]
        result[i] = s
    return result


def test_spmv_basic() -> None:
    # Simple 3x3 matrix with a few non‑zeros
    dense = [
        [0.0, 2.0, 0.0],
        [3.0, 0.0, 4.0],
        [0.0, 0.0, 5.0],
    ]
    csr = CSRMatrix(
        data=[2.0, 3.0, 4.0, 5.0],
        indices=[1, 0, 2, 2],
        indptr=[0, 1, 3, 4],
        shape=(3, 3),
    )
    x = [1.0, 2.0, 3.0]
    expected = dense_spmv_dense(dense, x)
    result = spmv(csr, x)
    assert result == expected, f"Basic test failed: {result} != {expected}"


def test_spmv_empty() -> None:
    csr = CSRMatrix(data=[], indices=[], indptr=[0, 0, 0], shape=(2, 3))
    x = [0.0, 0.0, 0.0]
    result = spmv(csr, x)
    assert result == [0.0, 0.0], "Empty matrix should produce zero vector"


def test_spmv_single_element() -> None:
    csr = CSRMatrix(data=[7.5], indices=[2], indptr=[0, 0, 1], shape=(2, 4))
    x = [1.0, 2.0, 3.0, 4.0]
    # Matrix:
    # [0 0 0 0]
    # [0 0 7.5 0]
    expected = [0.0, 7.5 * x[2]]
    result = spmv(csr, x)
    assert result == expected, "Single element multiplication incorrect"


def test_spmv_dimension_mismatch() -> None:
    csr = CSRMatrix(data=[1.0], indices=[0], indptr=[0, 1], shape=(1, 1))
    x = [1.0, 2.0]  # Too long
    try:
        spmv(csr, x)
    except ValueError as e:
        assert "Dimension mismatch" in str(e)
    else:
        assert False, "Expected ValueError for dimension mismatch"


def run_all_tests() -> None:
    test_spmv_basic()
    test_spmv_empty()
    test_spmv_single_element()
    test_spmv_dimension_mismatch()
    print("All unit tests passed.")


def demo() -> None:
    # Demonstrate with a 4x5 matrix
    dense = [
        [0, 1, 0, 0, 2],
        [3, 0, 4, 0, 0],
        [0, 0, 0, 5, 0],
        [6, 0, 0, 0, 7],
    ]
    csr = CSRMatrix(
        data=[1, 2, 3, 4, 5, 6, 7],
        indices=[1, 4, 0, 2, 3, 0, 4],
        indptr=[0, 2, 4, 5, 7],
        shape=(4, 5),
    )
    x = [1, 2, 3, 4, 5]
    y = spmv(csr, x)
    expected = dense_spmv_dense(dense, x)
    print("Sparse matrix‑vector multiplication result:", y)
    print("Expected (dense) result:", expected)


if __name__ == "__main__":
    run_all_tests()
    demo()
