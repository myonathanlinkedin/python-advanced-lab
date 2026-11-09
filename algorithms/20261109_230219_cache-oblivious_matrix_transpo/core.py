from __future__ import annotations
from typing import List, Sequence, Any

Matrix = List[List[Any]]

def _create_matrix(rows: int, cols: int, fill: Any = 0) -> Matrix:
    """Create a rows×cols matrix filled with `fill`."""
    return [[fill for _ in range(cols)] for _ in range(rows)]

def _transpose_recursive(
    src: Matrix,
    dst: Matrix,
    src_row: int,
    src_col: int,
    dst_row: int,
    dst_col: int,
    rows: int,
    cols: int,
    *,
    threshold: int = 16,
) -> None:
    """
    Recursively transpose a sub‑matrix of `src` into `dst`.

    The sub‑matrix starts at (src_row, src_col) with size rows×cols.
    Its transposed counterpart is placed at (dst_row, dst_col) in `dst`.
    """
    if rows <= threshold and cols <= threshold:
        # Base case: copy element‑wise.
        for i in range(rows):
            src_i = src_row + i
            dst_i = dst_row + i
            src_row_data = src[src_i]
            dst_row_data = dst[dst_i]
            for j in range(cols):
                dst_row_data[dst_col + j] = src_row_data[src_col + j]
        return

    # Split the larger dimension.
    if rows >= cols:
        mid = rows // 2
        # Upper part
        _transpose_recursive(
            src, dst,
            src_row, src_col,
            dst_row, dst_col,
            mid, cols,
            threshold=threshold,
        )
        # Lower part
        _transpose_recursive(
            src, dst,
            src_row + mid, src_col,
            dst_row, dst_col + mid,
            rows - mid, cols,
            threshold=threshold,
        )
    else:
        mid = cols // 2
        # Left part
        _transpose_recursive(
            src, dst,
            src_row, src_col,
            dst_row, dst_col,
            rows, mid,
            threshold=threshold,
        )
        # Right part
        _transpose_recursive(
            src, dst,
            src_row, src_col + mid,
            dst_row + mid, dst_col,
            rows, cols - mid,
            threshold=threshold,
        )

def transpose(matrix: Matrix) -> Matrix:
    """
    Return the transpose of `matrix` using a cache‑oblivious divide‑and‑conquer
    algorithm.

    The algorithm works for any rectangular matrix and runs in O(m·n) time,
    using O(log(min(m,n))) recursion depth and no explicit cache‑size
    parameters.
    """
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    # Validate rectangular shape.
    for row in matrix:
        if len(row) != cols:
            raise ValueError("All rows must have the same length.")
    result = _create_matrix(cols, rows)
    _transpose_recursive(matrix, result, 0, 0, 0, 0, rows, cols)
    return result
