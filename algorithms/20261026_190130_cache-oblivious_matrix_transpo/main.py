from __future__ import annotations
import random
import sys
from typing import List, Any
from .types import Matrix, T
from .engine import cache_oblivious_transpose

def _naive_transpose(matrix: Matrix[T]) -> Matrix[T]:
    """Reference implementation using Python's built‑in idioms."""
    if not matrix:
        return []
    return [list(col) for col in zip(*matrix)]

def _generate_random_matrix(rows: int, cols: int, seed: int = 0) -> Matrix[int]:
    rnd = random.Random(seed)
    return [[rnd.randint(-1000, 1000) for _ in range(cols)] for _ in range(rows)]

def _assert_matrices_equal(a: Matrix[Any], b: Matrix[Any]) -> None:
    assert len(a) == len(b), f"Row count mismatch: {len(a)} vs {len(b)}"
    for i, (ra, rb) in enumerate(zip(a, b)):
        assert len(ra) == len(rb), f"Column count mismatch at row {i}: {len(ra)} vs {len(rb)}"
        for j, (va, vb) in enumerate(zip(ra, rb)):
            assert va == vb, f"Value mismatch at ({i},{j}): {va} != {vb}"

def _run_tests() -> None:
    # Edge cases.
    empty: Matrix[int] = []
    assert cache_oblivious_transpose(empty) == []

    single_row = [[1, 2, 3, 4]]
    expected = [[1], [2], [3], [4]]
    _assert_matrices_equal(cache_oblivious_transpose(single_row), expected)

    single_col = [[1], [2], [3], [4]]
    expected = [[1, 2, 3, 4]]
    _assert_matrices_equal(cache_oblivious_transpose(single_col), expected)

    # Square matrices of varying sizes.
    for n in [1, 2, 3, 4, 5, 8, 16, 31, 64]:
        mat = _generate_random_matrix(n, n, seed=n)
        trans1 = cache_oblivious_transpose(mat)
        trans2 = _naive_transpose(mat)
        _assert_matrices_equal(trans1, trans2)

    # Rectangular matrices.
    sizes = [(2, 5), (5, 2), (3, 7), (7, 3), (10, 15), (15, 10), (1, 0), (0, 1)]
    for rows, cols in sizes:
        mat = _generate_random_matrix(rows, cols, seed=rows * 100 + cols)
        trans1 = cache_oblivious_transpose(mat)
        trans2 = _naive_transpose(mat)
        _assert_matrices_equal(trans1, trans2)

    # Larger random matrices (still deterministic).
    for rows, cols in [(64, 48), (128, 64), (64, 128), (100, 100)]:
        mat = _generate_random_matrix(rows, cols, seed=rows + cols)
        trans1 = cache_oblivious_transpose(mat)
        trans2 = _naive_transpose(mat)
        _assert_matrices_equal(trans1, trans2)

    print("All unit tests passed.")

def _demo() -> None:
    """Simple interactive demo."""
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    print("Original matrix:")
    for row in matrix:
        print(row)
    transposed = cache_oblivious_transpose(matrix)
    print("\nTransposed matrix:")
    for row in transposed:
        print(row)

if __name__ == "__main__":
    _run_tests()
    _demo()
