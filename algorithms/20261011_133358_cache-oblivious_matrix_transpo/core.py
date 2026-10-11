from typing import List, TypeVar

T = TypeVar('T')


def cache_oblivious_transpose(matrix: List[List[T]]) -> List[List[T]]:
    """
    Return the transpose of *matrix* using a cache‑oblivious recursive algorithm.
    The input matrix must be rectangular (all rows same length). The original
    matrix is not modified.

    Parameters
    ----------
    matrix: List[List[T]]
        A 2‑D list representing the matrix to transpose.

    Returns
    -------
    List[List[T]]
        The transposed matrix.
    """
    if not matrix:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    # Validate rectangular shape
    for r in matrix:
        if len(r) != cols:
            raise ValueError("All rows must have the same length")

    # Allocate destination matrix (cols x rows) filled with placeholders
    dst: List[List[T]] = [[None for _ in range(rows)] for _ in range(cols)]

    _transpose_recursive(
        src=matrix,
        dst=dst,
        src_r=0,
        src_c=0,
        dst_r=0,
        dst_c=0,
        rows=rows,
        cols=cols,
        leaf_size=64,
    )
    return dst


def _transpose_recursive(
    src: List[List[T]],
    dst: List[List[T]],
    src_r: int,
    src_c: int,
    dst_r: int,
    dst_c: int,
    rows: int,
    cols: int,
    leaf_size: int,
) -> None:
    """
    Recursively transpose a sub‑matrix of size (rows × cols) from *src* into *dst*.
    The element at src[src_r + i][src_c + j] is placed at dst[dst_r + j][dst_c + i].

    The recursion stops when the sub‑matrix contains at most *leaf_size* elements,
    at which point a simple double loop copies the data.
    """
    if rows == 0 or cols == 0:
        return

    if rows * cols <= leaf_size:
        # Base case: naive copy
        for i in range(rows):
            src_row = src[src_r + i]
            for j in range(cols):
                dst[dst_r + j][dst_c + i] = src_row[src_c + j]
        return

    # Choose split dimension: split the larger side to keep sub‑problems balanced
    if rows >= cols:
        mid = rows // 2
        # Top half of src ↔ left half of dst (columns)
        _transpose_recursive(
            src,
            dst,
            src_r,
            src_c,
            dst_r,
            dst_c,
            mid,
            cols,
            leaf_size,
        )
        # Bottom half of src ↔ right half of dst
        _transpose_recursive(
            src,
            dst,
            src_r + mid,
            src_c,
            dst_r,
            dst_c + mid,
            rows - mid,
            cols,
            leaf_size,
        )
    else:
        mid = cols // 2
        # Left half of src ↔ top half of dst (rows)
        _transpose_recursive(
            src,
            dst,
            src_r,
            src_c,
            dst_r,
            dst_c,
            rows,
            mid,
            leaf_size,
        )
        # Right half of src ↔ bottom half of dst
        _transpose_recursive(
            src,
            dst,
            src_r,
            src_c + mid,
            dst_r + mid,
            dst_c,
            rows,
            cols - mid,
            leaf_size,
        )
