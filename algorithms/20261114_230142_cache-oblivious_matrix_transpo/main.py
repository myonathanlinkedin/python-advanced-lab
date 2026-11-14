import random
import time
from typing import List, Any

from core import transpose

def _generate_matrix(rows: int, cols: int, seed: int | None = None) -> List[List[int]]:
    rng = random.Random(seed)
    return [[rng.randint(0, 1000) for _ in range(cols)] for _ in range(rows)]

def _assert_equal(a: List[List[Any]], b: List[List[Any]]) -> None:
    assert a == b, f"Expected {a} == {b}"

def test_empty() -> None:
    assert transpose([]) == []

def test_single_row() -> None:
    m = [[1, 2, 3]]
    expected = [[1], [2], [3]]
    _assert_equal(transpose(m), expected)

def test_single_col() -> None:
    m = [[1], [2], [3]]
    expected = [[1, 2, 3]]
    _assert_equal(transpose(m), expected)

def test_square() -> None:
    m = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    expected = [[1, 4, 7],
                [2, 5, 8],
                [3, 6, 9]]
    _assert_equal(transpose(m), expected)

def test_random() -> None:
    m = _generate_matrix(5, 7, seed=42)
    t = transpose(m)
    # Verify dimensions
    assert len(t) == 7 and all(len(row) == 5 for row in t)
    # Verify element-wise
    for i in range(5):
        for j in range(7):
            assert t[j][i] == m[i][j]

def test_inverse() -> None:
    m = _generate_matrix(8, 3, seed=99)
    t = transpose(m)
    tt = transpose(t)
    _assert_equal(tt, m)

def test_non_rectangular() -> None:
    m = [[1, 2], [3]]
    try:
        transpose(m)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for non-rectangular matrix"

def benchmark(rows: int = 512, cols: int = 512) -> None:
    m = _generate_matrix(rows, cols, seed=123)
    start = time.perf_counter()
    t = transpose(m)
    elapsed = time.perf_counter() - start
    print(f"Transposed {rows}x{cols} matrix in {elapsed:.4f} seconds")

def run_tests() -> None:
    test_empty()
    test_single_row()
    test_single_col()
    test_square()
    test_random()
    test_inverse()
    test_non_rectangular()
    print("All tests passed.")

if __name__ == "__main__":
    run_tests()
    print("\nDemo: Transpose a 4x3 matrix")
    demo = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]]
    print("Original:")
    for row in demo:
        print(row)
    transposed = transpose(demo)
    print("\nTransposed:")
    for row in transposed:
        print(row)
    print("\nBenchmarking 512x512 matrix:")
    benchmark()
