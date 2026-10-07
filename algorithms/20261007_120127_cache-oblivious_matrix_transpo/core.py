from __future__ import annotations

from typing import List, Sequence

Matrix = List[List[int]]  # Generic numeric matrix; can be any type supporting assignment


def _make_matrix(rows: int, cols: int, fill: int = 0) -> Matrix:
    """Create a rows×cols matrix filled with `fill`."""
    return [[fill for _ in range(cols)] for _ in range(rows)]


def _transpose_recursive(
    src: Sequence[Sequence[int]],
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
    Recursively transpose a sub‑matrix.

    src  – source matrix (read‑only)
    dst  – destination matrix (write‑only)
    (src_row, src_col) – top‑left corner in src
    (dst_row, dst_col) – top‑left corner in dst
    rows, cols – dimensions of the sub‑matrix to transpose
    threshold – when rows*cols ≤ threshold, perform a simple copy.
    """
    if rows == 0 or cols == 0:
        return
    if rows * cols <= threshold:
        # Base case: direct copy
        for i in range(rows):
            src_i = src_row + i
            dst_i = dst_col + i  # note the swap of row/col for destination
            src_row_data = src[src_i]
            for j in range(cols):
                dst[dst_row + j][dst_i] = src_row_data[src_col + j]
        return

    # Divide the larger dimension
    if rows >= cols:
        mid = rows // 2
        _transpose_recursive(src, dst,
                             src_row, src_col,
                             dst_row, dst_col,
                             mid, cols, threshold)
        _transpose_recursive(src, dst,
                             src_row + mid, src_col,
                             dst_row, dst_col + mid,
                             rows - mid, cols, threshold)
    else:
        mid = cols // 2
        _transpose_recursive(src, dst,
                             src_row, src_col,
                             dst_row, dst_col,
                             rows, mid, threshold)
        _transpose_recursive(src, dst,
                             src_row, src_col + mid,
                             dst_row + mid, dst_col,
                             rows, cols - mid, threshold)


def transpose(matrix: List[List[int]], threshold: int = 32) -> Matrix:
    """
    Return the transpose of `matrix` using a cache‑oblivious divide‑and‑conquer algorithm.

    Parameters
    ----------
    matrix : List[List[int]]
        Input 2‑D list. All rows must have the same length.
    threshold : int, optional
        Base‑case size; smaller sub‑matrices are copied directly.
        Default is 32 (empirically good for many caches).

    Returns
    -------
    List[List[int]]
        New matrix with dimensions swapped.
    """
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    # Validate rectangular shape
    for r in matrix:
        if len(r) != cols:
            raise ValueError("All rows must have the same length")
    dst = _make_matrix(cols, rows)
    _transpose_recursive(matrix, dst, 0, 0, 0, 0, rows, cols, threshold)
    return dst
