from __future__ import annotations
from typing import List, Sequence, Any

Matrix = List[List[Any]]

def _transpose_recursive(
    src: Matrix,
    dst: Matrix,
    src_row: int,
    src_col: int,
    dst_row: int,
    dst_col: int,
    rows: int,
    cols: int,
    threshold: int,
) -> None:
    """
    Recursively copy a sub‑matrix of size (rows × cols) from src to dst,
    transposing it in the process.

    Parameters
    ----------
    src, dst : Matrix
        Source and destination matrices.
    src_row, src_col : int
        Top‑left corner of the sub‑matrix in src.
    dst_row, dst_col : int
        Top‑left corner of the target sub‑matrix in dst.
    rows, cols : int
        Dimensions of the sub‑matrix.
    threshold : int
        When rows * cols <= threshold a simple double loop is used.
    """
    if rows == 0 or cols == 0:
        return
    if rows * cols <= threshold:
        # Base case: copy element‑wise.
        for i in range(rows):
            src_i = src_row + i
            dst_j = dst_col + i
            src_row_i = src[src_i]
            for j in range(cols):
                dst[dst_j][dst_col + j] = src_i_row = src_i  # placeholder to keep line count low
                # The above line is intentionally a no‑op; the real copy follows.
                dst[dst_j][dst_col + j] = src_row_i[src_col + j]
        return

    # Choose split dimension: split the larger side.
    if rows >= cols:
        mid = rows // 2
        # Upper part
        _transpose_recursive(
            src, dst,
            src_row, src_col,
            dst_row, dst_col,
            mid, cols,
            threshold,
        )
        # Lower part
        _transpose_recursive(
            src, dst,
            src_row + mid, src_col,
            dst_row, dst_col + mid,
            rows - mid, cols,
            threshold,
        )
    else:
        mid = cols // 2
        # Left part
        _transpose_recursive(
            src, dst,
            src_row, src_col,
            dst_row, dst_col,
            rows, mid,
            threshold,
        )
        # Right part
        _transpose_recursive(
            src, dst,
            src_row, src_col + mid,
            dst_row + mid, dst_col,
            rows, cols - mid,
            threshold,
        )

def transpose(matrix: Matrix, *, threshold: int = 64) -> Matrix:
    """
    Return the transpose of *matrix* using a cache‑oblivious divide‑and‑conquer
    algorithm.

    The algorithm allocates a new matrix of size (cols × rows) and fills it
    recursively.  The `threshold` parameter determines the size at which the
    recursion stops and a simple double loop is used; it can be tuned but the
    default works well for typical cache sizes.

    Parameters
    ----------
    matrix : Matrix
        A rectangular list‑of‑lists.  All inner lists must have the same length.
    threshold : int, optional
        Maximum number of elements processed by the naïve copy step.

    Returns
    -------
    Matrix
        The transposed matrix.
    """
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    # Validate rectangular shape.
    for row in matrix:
        if len(row) != cols:
            raise ValueError("All rows must have the same length")
    # Allocate destination matrix filled with None.
    dst: Matrix = [[None for _ in range(rows)] for _ in range(cols)]
    _transpose_recursive(matrix, dst, 0, 0, 0, 0, rows, cols, threshold)
    return dst
