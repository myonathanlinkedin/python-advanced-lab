from __future__ import annotations
import sys
from typing import List
from core import transpose

def _assert_equal(a: List[List[int]], b: List[List[int]]) -> None:
    assert a == b, f"Expected {b}, got {a}"

def _test_identity() -> None:
    m = [[1]]
    t = transpose(m)
    _assert_equal(t, [[1]])
    _assert_equal(m, [[1]])  # original unchanged

def _test_small_square() -> None:
    m = [[1, 2], [3, 4]]
    t = transpose(m)
    _assert_equal(t, [[1, 3], [2, 4]])

def _test_odd_size() -> None:
    m = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    t = transpose(m)
    _assert_equal(t, [[1, 4, 7], [2, 5, 8], [3, 6, 9]])

def _test_even_size() -> None:
    m = [[i + j * 4 + 1 for i in range(4)] for j in range(4)]
    expected = [[m[j][i] for j in range(4)] for i in range(4)]
    t = transpose(m)
    _assert_equal(t, expected)

def _test_large_size() -> None:
    n = 7
    m = [[i + j * n + 1 for i in range(n)] for j in range(n)]
    expected = [[m[j][i] for j in range(n)] for i in range(n)]
    t = transpose(m)
    _assert_equal(t, expected)

def _test_double_transpose() -> None:
    m = [[i + j * 5 + 1 for i in range(5)] for j in range(5)]
    t = transpose(m)
    tt = transpose(t)
    _assert_equal(tt, m)

def _test_non_square() -> None:
    m = [[1, 2, 3], [4, 5, 6]]
    try:
        transpose(m)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-square matrix")

def run_tests() -> None:
    _test_identity()
    _test_small_square()
    _test_odd_size()
    _test_even_size()
    _test_large_size()
    _test_double_transpose()
    _test_non_square()
    print("All tests passed.")

if __name__ == '__main__':
    run_tests()
    # Demo: transpose a 4x4 matrix
    demo = [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9,10,11,12],
            [13,14,15,16]]
    print("Original:")
    for row in demo:
        print(row)
    transposed = transpose(demo)
    print("\nTransposed:")
    for row in transposed:
        print(row)
