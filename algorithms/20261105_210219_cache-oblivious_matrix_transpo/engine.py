from __future__ import annotations

from typing import List, TypeVar, Generic, Callable
from .types import Matrix, T

def _transpose_direct(src: Matrix[T], dst: Matrix[T],
                      src_row: int, src_col: int,
                      dst_row: int, dst_col: int,
                      rows: int, cols: int) -> None:
    """
    Direct (non‑recursive) copy of a sub‑matrix.
    Copies src[src_row:src_row+rows][src_col:src_col+cols] into
    dst[dst_row:dst_row+cols][dst_col:dst_col+rows] transposed.
    """
    for i in range(rows):
        src_r = src_row + i
        dst_c = dst_col + i
        src_row_i = src[src_r]
        for j in range(cols):
            dst[dst_row + j][dst_c] = src_row_i[src_col + j]

def _transpose_recursive(src: Matrix[T], dst: Matrix[T],
                         src_row: int, src_col: int,
                         dst_row: int, dst_col: int,
                         rows: int, cols: int,
                         leaf: int = 16) -> None:
    """
    Cache‑oblivious recursive transposition.
    Copies the sub‑matrix defined by (src_row, src_col, rows, cols)
    from src into dst transposed, starting at (dst_row, dst_col).
    """
    if rows <= leaf or cols <= leaf:
        _transpose_direct(src, dst,
                          src_row, src_col,
                          dst_row, dst_col,
                          rows, cols)
        return

    if rows >= cols:
        # Split rows
        mid = rows // 2
        _transpose_recursive(src, dst,
                             src_row, src_col,
                             dst_row, dst_col,
                             mid, cols, leaf)
        _transpose_recursive(src, dst,
                             src_row + mid, src_col,
                             dst_row, dst_col + mid,  # note swapped indices
                             rows - mid, cols, leaf)
    else:
        # Split columns
        mid = cols // 2
        _transpose_recursive(src, dst,
                             src_row, src_col,
                             dst_row, dst_col,
                             rows, mid, leaf)
        _transpose_recursive(src, dst,
                             src_row, src_col + mid,
                             dst_row + mid, dst_col,
                             rows, cols - mid, leaf)

def transpose(matrix: Matrix[T]) -> Matrix[T]:
    """
    Public API: returns a new matrix that is the transpose of *matrix*.
    The implementation uses a cache‑oblivious divide‑and‑conquer algorithm.
    """
    if not matrix:
        return []

    row_count = len(matrix)
    col_count = len(matrix[0])

    # Validate rectangular shape
    for r in matrix:
        if len(r) != col_count:
            raise ValueError("All rows must have the same length")

    # Allocate destination matrix filled with None placeholders
    dst: Matrix[T] = [[None for _ in range(row_count)] for _ in range(col_count)]  # type: ignore

    _transpose_recursive(matrix, dst,
                         0, 0, 0, 0,
                         row_count, col_count)

    return dst
