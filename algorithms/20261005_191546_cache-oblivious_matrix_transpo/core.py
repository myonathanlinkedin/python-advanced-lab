"""
Cache‑Oblivious Matrix Transposition
===================================

Provides a high‑performance, cache‑oblivious implementation of matrix
transposition using a divide‑and‑conquer strategy. The algorithm works
for any rectangular matrix and falls back to a simple nested‑loop copy
when the sub‑matrix size falls below a tunable threshold.

Only the Python standard library is used.
"""

from __future__ import annotations

from typing import List, Sequence, TypeVar

T = TypeVar("T")

# Tunable base case size – experimentally a small constant works well.
_BASE_CASE_THRESHOLD = 64  # number of elements (rows * cols) per leaf


def _naive_transpose(src: Sequence[Sequence[T]]) -> List[List[T]]:
    """Simple O(m·n) transpose used for the base case."""
    if not src:
        return []
    rows, cols = len(src), len(src[0])
    return [[src[i][j] for i in range(rows)] for j in range(cols)]


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
    Recursively transpose a sub‑matrix of size ``rows × cols`` from ``src``
    into ``dst``. ``src`` is accessed at (src_row + i, src_col + j) and the
    result is written to (dst_row + j, dst_col + i) in ``dst``.
    """
    # Base case: use naive copy when the sub‑matrix is small enough.
    if rows * cols <= _BASE_CASE_THRESHOLD:
        for i in range(rows):
            src_i = src_row + i
            src_row_i = src[src_i]
            dst_i = dst_row + i
            for j in range(cols):
                dst[dst_col + i][dst_row + j] = src_row_i[src_col + j]
        return

    # Split the larger dimension to keep sub‑problems balanced.
    if rows >= cols:
        mid = rows // 2
        # Top half
        _transpose_recursive(
            src,
            dst,
            src_row,
            src_col,
            dst_row,
            dst_col,
            mid,
            cols,
        )
        # Bottom half
        _transpose_recursive(
            src,
            dst,
            src_row + mid,
            src_col,
            dst_row,
            dst_col + mid,
            rows - mid,
            cols,
        )
    else:
        mid = cols // 2
        # Left half
        _transpose_recursive(
            src,
            dst,
            src_row,
            src_col,
            dst_row,
            dst_col,
            rows,
            mid,
        )
        # Right half
        _transpose_recursive(
            src,
            dst,
            src_row,
            src_col + mid,
            dst_row + mid,
            dst_col,
            rows,
            cols - mid,
        )


def transpose(matrix: Sequence[Sequence[T]]) -> List[List[T]]:
    """
    Return the transpose of ``matrix`` using a cache‑oblivious algorithm.

    Parameters
    ----------
    matrix:
        A rectangular 2‑D sequence (list of lists, tuple of tuples, …).

    Returns
    -------
    List[List[T]]
        New matrix with dimensions swapped.
    """
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    # Allocate destination matrix filled with placeholders.
    dst: List[List[T]] = [[None for _ in range(rows)] for _ in range(cols)]  # type: ignore[var-annotated]

    _transpose_recursive(matrix, dst, 0, 0, 0, 0, rows, cols)
    return dst


def transpose_inplace(square: List[List[T]]) -> None:
    """
    In‑place transpose for a square matrix using a cache‑oblivious approach.
    The matrix is modified directly.

    Parameters
    ----------
    square:
        A mutable square matrix (list of lists). Must be ``n × n``.
    """
    n = len(square)
    if any(len(row) != n for row in square):
        raise ValueError("In‑place transpose requires a square matrix.")
    # In‑place version works by swapping symmetric elements.
    # Recursively process quadrants to improve cache behaviour.
    def recurse(r0: int, c0: int, size: int) -> None:
        if size <= 1:
            return
        half = size // 2
        # Process four sub‑quadrants
        recurse(r0, c0, half)
        recurse(r0, c0 + half, half)
        recurse(r0 + half, c0, half)
        recurse(r0 + half, c0 + half, half)
        # Swap the off‑diagonal blocks
        for i in range(half):
            for j in range(half):
                a_i, a_j = r0 + i, c0 + half + j
                b_i, b_j = r0 + half + j, c0 + i
                square[a_i][a_j], square[b_i][b_j] = square[b_i][b_j], square[a_i][a_j]

    recurse(0, 0, n)


# Public API
__all__ = ["transpose", "transpose_inplace"]
