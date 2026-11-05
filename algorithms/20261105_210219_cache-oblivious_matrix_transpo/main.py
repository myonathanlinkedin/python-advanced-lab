from __future__ import annotations

from typing import List, Any
from .engine import transpose
from .types import Matrix

def _naive_transpose(matrix: Matrix[Any]) -> Matrix[Any]:
    """Reference implementation using Python's built‑in constructs."""
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]

def _test_empty() -> None:
    assert transpose([]) == []

def _test_square() -> None:
    m = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    expected = _naive_transpose(m)
    result = transpose(m)
    assert result == expected

def _test_rectangular_wide() -> None:
    m = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    expected = _naive_transpose(m)
    result = transpose(m)
    assert result == expected

def _test_rectangular_tall() -> None:
    m = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ]
    expected = _naive_transpose(m)
    result = transpose(m)
    assert result == expected

def _test_single_row() -> None:
    m = [[1, 2, 3, 4, 5]]
    expected = _naive_transpose(m)
    result = transpose(m)
    assert result == expected

def _test_single_column() -> None:
    m = [[1], [2], [3], [4]]
    expected = _naive_transpose(m)
    result = transpose(m)
    assert result == expected

def _test_non_rectangular_error() -> None:
    m = [
        [1, 2, 3],
        [4, 5],
    ]
    try:
        transpose(m)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for non‑rectangular matrix"

def _run_all_tests() -> None:
    _test_empty()
    _test_square()
    _test_rectangular_wide()
    _test_rectangular_tall()
    _test_single_row()
    _test_single_column()
    _test_non_rectangular_error()
    print("All cache‑oblivious transpose tests passed.")

if __name__ == '__main__':
    _run_all_tests()
