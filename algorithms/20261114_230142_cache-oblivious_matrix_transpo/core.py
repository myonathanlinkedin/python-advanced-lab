from __future__ import annotations
from typing import List, TypeVar, Any

T = TypeVar('T')

__all__ = ['transpose']

def transpose(matrix: List[List[T]]) -> List[List[T]]:
    """
    Return the transpose of a 2D matrix using a cache-oblivious divide-and-conquer algorithm.
    The input matrix must be rectangular (all rows of equal length). An empty matrix
    (zero rows or zero columns) is returned unchanged.

    Parameters
    ----------
    matrix : List[List[T]]
        The source matrix.

    Returns
    -------
    List[List[T]]
        The transposed matrix.

    Raises
    ------
    ValueError
        If the input matrix is not rectangular.
    """
    if not matrix:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    # Validate rectangularity
    for row in matrix:
        if len(row) != cols:
            raise ValueError("All rows must have the same length")

    if rows == 0 or cols == 0:
        return []

    # Allocate destination matrix
    result: List[List[T]] = [[None] * rows for _ in range(cols)]

    _transpose_rec(matrix, result,
                   0, 0, rows, cols,
                   0, 0)
    return result

def _transpose_rec(src: List[List[T]],
                   dst: List[List[T]],
                   r0: int, c0: int, r1: int, c1: int,
                   dr0: int, dc0: int,
                   threshold: int = 64) -> None:
    """
    Recursive helper that transposes a submatrix of src into a submatrix of dst.
    The submatrix in src is rows [r0:r1) and cols [c0:c1).
    The corresponding submatrix in dst is rows [dc0:dc0+(c1-c0)) and cols [dr0:dr0+(r1-r0)).
    """
    rows = r1 - r0
    cols = c1 - c0

    if rows <= threshold or cols <= threshold:
        for i in range(rows):
            src_row = src[r0 + i]
            dst_col = dst[dc0]
            for j in range(cols):
                dst[dc0 + j][dr0 + i] = src_row[c0 + j]
        return

    mid_r = r0 + rows // 2
    mid_c = c0 + cols // 2
    half_rows = rows // 2
    half_cols = cols // 2

    # Top-left quadrant
    _transpose_rec(src, dst,
                   r0, c0, mid_r, mid_c,
                   dr0, dc0,
                   threshold)

    # Top-right -> bottom-left
    _transpose_rec(src, dst,
                   r0, mid_c, mid_r, c1,
                   dr0, dc0 + half_cols,
                   threshold)

    # Bottom-left -> top-right
    _transpose_rec(src, dst,
                   mid_r, c0, r1, mid_c,
                   dr0 + half_rows, dc0,
                   threshold)

    # Bottom-right
    _transpose_rec(src, dst,
                   mid_r, mid_c, r1, c1,
                   dr0 + half_rows, dc0 + half_cols,
                   threshold)
