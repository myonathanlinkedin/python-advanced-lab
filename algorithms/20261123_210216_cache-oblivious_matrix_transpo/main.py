from __future__ import annotations
from typing import List, Any
import sys
import random

# Import from sibling modules.
from custom_types import Matrix
from engine import transpose

def _naive_transpose(matrix: Matrix[Any]) -> Matrix[Any]:
    """Reference implementation using a straightforward double loop."""
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]

def _assert_equal(m1: Matrix[Any], m2: Matrix[Any]) -> None:
    assert len(m1) == len(m2), f"Row count mismatch: {len(m1)} vs {len(m2)}"
    for i, (row1, row2) in enumerate(zip(m1, m2)):
        assert len(row1) == len(row2), f"Column count mismatch at row {i}"
        for j, (a, b) in enumerate(zip(row1, row2)):
            assert a == b, f"Element mismatch at ({i}, {j}): {a} != {b}"

def _run_tests() -> None:
    # 1. Empty matrix.
    empty: Matrix[int] = []
    assert transpose(empty) == [], "Transpose of empty matrix failed"

    # 2. Single element.
    single = [[42]]
    assert transpose(single) == [[42]], "Transpose of 1×1 matrix failed"

    # 3. Square matrix with power‑of‑two dimensions.
    size = 8
    sq = [[i * size + j for j in range(size)] for i in range(size)]
    _assert_equal(transpose(sq), _naive_transpose(sq))

    # 4. Square matrix with odd dimension.
    size = 5
    sq_odd = [[i * size + j for j in range(size)] for i in range(size)]
    _assert_equal(transpose(sq_odd), _naive_transpose(sq_odd))

    # 5. Rectangular matrix (more rows than columns).
    rows, cols = 7, 3
    rect1 = [[i * cols + j for j in range(cols)] for i in range(rows)]
    _assert_equal(transpose(rect1), _naive_transpose(rect1))

    # 6. Rectangular matrix (more columns than rows).
    rows, cols = 3, 9
    rect2 = [[i * cols + j for j in range(cols)] for i in range(rows)]
    _assert_equal(transpose(rect2), _naive_transpose(rect2))

    # 7. Randomized matrices of various sizes.
    for rows in range(1, 13):
        for cols in range(1, 13):
            mat = [[random.randint(-1000, 1000) for _ in range(cols)] for _ in range(rows)]
            _assert_equal(transpose(mat), _naive_transpose(mat))

    # 8. Verify that transposing twice yields the original matrix.
    for rows, cols in [(4, 6), (6, 4), (10, 10)]:
        mat = [[random.random() for _ in range(cols)] for _ in range(rows)]
        double = transpose(transpose(mat))
        _assert_equal(double, mat)

    print("All unit tests passed.")

def _demo() -> None:
    """Simple interactive demo printed to stdout."""
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    print("Original matrix:")
    for row in matrix:
        print(row)
    transposed = transpose(matrix)
    print("\nTransposed matrix:")
    for row in transposed:
        print(row)

if __name__ == "__main__":
    _run_tests()
    _demo()
