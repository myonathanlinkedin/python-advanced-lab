from __future__ import annotations
from typing import List, TypeVar, Generic, Sequence

T = TypeVar('T')

def transpose(matrix: Sequence[Sequence[T]]) -> List[List[T]]:
    """
    Cache‑oblivious matrix transposition.

    Returns a new matrix that is the transpose of *matrix*.
    The implementation uses a divide‑and‑conquer strategy that
    recursively splits the larger dimension until the sub‑problem
    fits into a small cache‑friendly base case.

    The algorithm works for any rectangular matrix, including empty
    matrices and matrices with a single row or column.

    Time   : O(m·n) where m = rows, n = cols
    Space  : O(m·n) for the output matrix plus O(log max(m,n)) recursion depth
    """
    if not matrix:
        return []

    rows = len(matrix)
    cols = len(matrix[0])
    # Verify rectangular shape
    for r in matrix:
        if len(r) != cols:
            raise ValueError("All rows must have the same length")

    # Allocate destination matrix
    result: List[List[T]] = [[None for _ in range(rows)] for _ in range(cols)]  # type: ignore[arg-type]

    _transpose_recursive(
        src=matrix,
        dst=result,
        src_row=0,
        src_col=0,
        dst_row=0,
        dst_col=0,
        rows=rows,
        cols=cols,
    )
    return result


# Threshold for the base case; chosen to fit typical L1 cache (~64 elements)
_BASE_CASE = 64

def _transpose_recursive(
    src: Sequence[Sequence[T]],
    dst: List[List[T]],
    src_row: int,
    src_col: int,
    dst_row: int,
    dst_col: int,
    rows: int,
    cols: int,
) -> None:
    """
    Recursively transpose a sub‑matrix of *src* into *dst*.

    Parameters
    ----------
    src, dst : matrices
    src_row, src_col : top‑left corner in *src*
    dst_row, dst_col : top‑left corner in *dst*
    rows, cols : dimensions of the sub‑matrix to transpose
    """
    # Base case: small enough to copy directly
    if rows * cols <= _BASE_CASE:
        for i in range(rows):
            src_i = src_row + i
            src_row_i = src[src_i]
            for j in range(cols):
                dst[dst_col + j][dst_row + i] = src_row_i[src_col + j]
        return

    # Recursive case: split the larger dimension
    if rows >= cols:
        # Split rows
        top = rows // 2
        bottom = rows - top
        # Transpose top part
        _transpose_recursive(
            src, dst,
            src_row, src_col,
            dst_row, dst_col,
            top, cols,
        )
        # Transpose bottom part
        _transpose_recursive(
            src, dst,
            src_row + top, src_col,
            dst_row, dst_col + top,
            bottom, cols,
        )
    else:
        # Split columns
        left = cols // 2
        right = cols - left
        # Transpose left part
        _transpose_recursive(
            src, dst,
            src_row, src_col,
            dst_row, dst_col,
            rows, left,
        )
        # Transpose right part
        _transpose_recursive(
            src, dst,
            src_row, src_col + left,
            dst_row + left, dst_col,
            rows, right,
        )
