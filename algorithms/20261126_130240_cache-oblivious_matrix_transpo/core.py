from __future__ import annotations
from typing import List, TypeVar, Generic

T = TypeVar('T')

def transpose(matrix: List[List[T]]) -> List[List[T]]:
    """
    Cache-oblivious matrix transposition for square matrices.
    Returns a new transposed matrix without modifying the input.
    Raises ValueError if the matrix is not square.
    """
    n = len(matrix)
    if n == 0:
        return []
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square")
    dst: List[List[T]] = [[None] * n for _ in range(n)]
    _transpose_recursive(matrix, dst, 0, 0, n)
    return dst

def _transpose_recursive(src: List[List[T]], dst: List[List[T]],
                         r0: int, c0: int, size: int) -> None:
    if size == 1:
        dst[c0][r0] = src[r0][c0]
    else:
        half = size // 2
        # top-left
        _transpose_recursive(src, dst, r0, c0, half)
        # top-right
        _transpose_recursive(src, dst, r0, c0 + half, half)
        # bottom-left
        _transpose_recursive(src, dst, r0 + half, c0, half)
        # bottom-right
        _transpose_recursive(src, dst, r0 + half, c0 + half, half)
        if size % 2 == 1:
            last = size - 1
            # last row
            for j in range(c0, c0 + last):
                dst[j][r0 + last] = src[r0 + last][j]
            # last column
            for i in range(r0, r0 + last):
                dst[c0 + last][i] = src[i][c0 + last]
            # corner
            dst[c0 + last][r0 + last] = src[r0 + last][c0 + last]
