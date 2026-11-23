from __future__ import annotations
from typing import List, TypeVar, Generic
from .types import Matrix, T

# Threshold for switching to the naive implementation.
# This value is chosen to fit typical L1 cache sizes; it can be tuned.
_BASE_CASE_SIZE = 64  # number of elements


def transpose(matrix: Matrix[T]) -> Matrix[T]:
    """
    Return the transpose of *matrix* using a cache‑oblivious recursive algorithm.

    The algorithm allocates a new matrix of size (cols × rows) and fills it
    by recursively copying sub‑blocks.  For sub‑matrices whose total element
    count does not exceed ``_BASE_CASE_SIZE`` a straightforward double loop
    is used.

    Parameters
    ----------
    matrix:
        A rectangular list‑of‑lists.  All inner lists must have the same length.

    Returns
    -------
    Matrix[T]
        The transposed matrix (cols × rows).

    Raises
    ------
    ValueError
        If the input matrix is ragged (rows of differing lengths).
    """
    if not matrix:
        return []  # empty matrix

    rows = len(matrix)
    cols = len(matrix[0])

    # Validate rectangular shape.
    for r in matrix:
        if len(r) != cols:
            raise ValueError("All rows must have the same number of columns")

    # Allocate destination matrix filled with placeholders.
    dst: Matrix[T] = [[None for _ in range(rows)] for _ in range(cols)]  # type: ignore[arg-type]

    _transpose_recursive(
        src=matrix,
        dst=dst,
        src_row=0,
        src_col=0,
        dst_row=0,
        dst_col=0,
        rows=rows,
        cols=cols,
    )
    return dst


def _transpose_recursive(
    src: Matrix[T],
    dst: Matrix[T],
    src_row: int,
    src_col: int,
    dst_row: int,
    dst_col: int,
    rows: int,
    cols: int,
) -> None:
    """
    Recursively copy a sub‑matrix of size (rows × cols) from *src* to *dst*,
    transposing it in the process.

    The recursion stops when the sub‑matrix contains at most
    ``_BASE_CASE_SIZE`` elements, at which point a simple double loop is used.
    """
    # Base case: small enough to copy directly.
    if rows * cols <= _BASE_CASE_SIZE:
        for i in range(rows):
            src_i = src_row + i
            for j in range(cols):
                src_j = src_col + j
                dst_j = dst_row + j
                dst_i = dst_col + i
                dst[dst_j][dst_i] = src[src_i][src_j]
        return

    # Recursive case: split the larger dimension.
    if rows >= cols:
        # Split rows roughly in half.
        top = rows // 2
        bottom = rows - top
        # Upper block
        _transpose_recursive(
            src,
            dst,
            src_row,
            src_col,
            dst_row,
            dst_col,
            top,
            cols,
        )
        # Lower block
        _transpose_recursive(
            src,
            dst,
            src_row + top,
            src_col,
            dst_row,
            dst_col + top,
            bottom,
            cols,
        )
    else:
        # Split columns roughly in half.
        left = cols // 2
        right = cols - left
        # Left block
        _transpose_recursive(
            src,
            dst,
            src_row,
            src_col,
            dst_row,
            dst_col,
            rows,
            left,
        )
        # Right block
        _transpose_recursive(
            src,
            dst,
            src_row,
            src_col + left,
            dst_row + left,
            dst_col,
            rows,
            right,
        )
