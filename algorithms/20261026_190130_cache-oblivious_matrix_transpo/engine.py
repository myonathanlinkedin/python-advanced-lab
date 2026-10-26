from __future__ import annotations
from typing import List, TypeVar, Generic
from .types import Matrix, T

# Threshold for switching to naive copy; small enough to fit in cache.
_BASE_CASE_SIZE = 16

def _naive_copy(
    src: Matrix[T],
    src_row: int,
    src_col: int,
    dst: Matrix[T],
    dst_row: int,
    dst_col: int,
    rows: int,
    cols: int,
) -> None:
    """Copy a sub‑matrix of size rows×cols from src to dst."""
    for i in range(rows):
        src_r = src_row + i
        dst_r = dst_row + i
        src_slice = src[src_r][src_col:src_col + cols]
        dst[dst_r][dst_col:dst_col + cols] = src_slice

def _transpose_recursive(
    src: Matrix[T],
    src_row: int,
    src_col: int,
    dst: Matrix[T],
    dst_row: int,
    dst_col: int,
    rows: int,
    cols: int,
) -> None:
    """
    Recursively transpose a sub‑matrix of `src` (rows×cols) into `dst`.
    The sub‑matrix of `src` starts at (src_row, src_col) and is written
    to `dst` starting at (dst_row, dst_col).  The algorithm is cache‑oblivious:
    it recursively splits the larger dimension until the sub‑problem fits
    into a small base case.
    """
    # Base case: small enough to copy directly.
    if rows * cols <= _BASE_CASE_SIZE:
        _naive_copy(src, src_row, src_col, dst, dst_row, dst_col, rows, cols)
        return

    if rows >= cols:
        # Split rows.
        mid = rows // 2
        # Top part.
        _transpose_recursive(
            src, src_row, src_col,
            dst, dst_row, dst_col,
            mid, cols,
        )
        # Bottom part.
        _transpose_recursive(
            src, src_row + mid, src_col,
            dst, dst_row + mid, dst_col,
            rows - mid, cols,
        )
    else:
        # Split columns.
        mid = cols // 2
        # Left part.
        _transpose_recursive(
            src, src_row, src_col,
            dst, dst_row, dst_col,
            rows, mid,
        )
        # Right part.
        _transpose_recursive(
            src, src_row, src_col + mid,
            dst, dst_row, dst_col + mid,
            rows, cols - mid,
        )

def cache_oblivious_transpose(matrix: Matrix[T]) -> Matrix[T]:
    """
    Return the transpose of `matrix` using a cache‑oblivious divide‑and‑conquer
    algorithm.  The input matrix is left unchanged.

    Time Complexity: Θ(m·n) where m = number of rows, n = number of columns.
    Space Complexity: Θ(m·n) for the output matrix plus O(log max(m,n)) recursion
    stack depth.
    """
    if not matrix:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    # Validate rectangular shape.
    for r in matrix:
        if len(r) != cols:
            raise ValueError("All rows must have the same length")

    # Allocate output matrix filled with placeholders.
    result: Matrix[T] = [[None for _ in range(rows)] for _ in range(cols)]  # type: ignore

    _transpose_recursive(matrix, 0, 0, result, 0, 0, rows, cols)
    return result
