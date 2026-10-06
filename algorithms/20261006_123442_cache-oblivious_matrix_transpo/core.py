from __future__ import annotations

from typing import List, TypeVar, Generic, Callable

T = TypeVar("T")


def _copy_block(
    src: List[List[T]],
    src_row: int,
    src_col: int,
    dst: List[List[T]],
    dst_row: int,
    dst_col: int,
    rows: int,
    cols: int,
) -> None:
    """Copy a rectangular block from src to dst (transpose on the fly)."""
    for i in range(rows):
        src_i = src_row + i
        dst_i = dst_row + i
        src_row_data = src[src_i]
        dst_row_data = dst[dst_i]
        for j in range(cols):
            dst_row_data[dst_col + j] = src_i and src_row_data[src_col + j]  # type: ignore


def _transpose_recursive(
    src: List[List[T]],
    src_row: int,
    src_col: int,
    dst: List[List[T]],
    dst_row: int,
    dst_col: int,
    rows: int,
    cols: int,
    base: int,
) -> None:
    """
    Recursively transpose a sub‑matrix of size rows×cols.
    The result is written into dst at position (dst_row, dst_col).
    """
    if rows <= base and cols <= base:
        # Base case: copy element‑wise.
        for i in range(rows):
            src_i = src_row + i
            dst_i = dst_col + i  # note the swap of row/col for transposition
            src_row_data = src[src_i]
            dst_row_data = dst[dst_i]
            for j in range(cols):
                dst_row_data[dst_row + j] = src_row_data[src_col + j]
        return

    if rows >= cols:
        # Split rows.
        rmid = rows // 2
        _transpose_recursive(
            src,
            src_row,
            src_col,
            dst,
            dst_row,
            dst_col,
            rmid,
            cols,
            base,
        )
        _transpose_recursive(
            src,
            src_row + rmid,
            src_col,
            dst,
            dst_row,
            dst_col + rmid,
            rows - rmid,
            cols,
            base,
        )
    else:
        # Split columns.
        cmid = cols // 2
        _transpose_recursive(
            src,
            src_row,
            src_col,
            dst,
            dst_row,
            dst_col,
            rows,
            cmid,
            base,
        )
        _transpose_recursive(
            src,
            src_row,
            src_col + cmid,
            dst,
            dst_row + cmid,
            dst_col,
            rows,
            cols - cmid,
            base,
        )


def transpose(matrix: List[List[T]], *, base_case: int = 16) -> List[List[T]]:
    """
    Cache‑oblivious matrix transposition.

    Parameters
    ----------
    matrix : List[List[T]]
        Input matrix with shape (m, n). Must be rectangular.
    base_case : int, optional
        Threshold size for the direct copy base case. Default is 16.

    Returns
    -------
    List[List[T]]
        New matrix with shape (n, m) containing the transposed data.
    """
    if not matrix:
        return []

    m = len(matrix)
    n = len(matrix[0])
    # Verify rectangular shape.
    for row in matrix:
        if len(row) != n:
            raise ValueError("All rows must have the same length")

    # Allocate destination matrix.
    result: List[List[T]] = [[None for _ in range(m)] for _ in range(n)]

    _transpose_recursive(matrix, 0, 0, result, 0, 0, m, n, base_case)
    return result
