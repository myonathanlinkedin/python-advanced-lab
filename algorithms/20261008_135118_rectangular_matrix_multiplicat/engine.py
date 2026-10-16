from __future__ import annotations

from typing import List
from .types import Matrix, MatrixDimensionError, MatrixStructureError, _validate_matrix


def multiply(A: Matrix, B: Matrix) -> Matrix:
    """
    Multiply two rectangular matrices A (m×k) and B (k×n) using the
    classic O(m·k·n) algorithm.

    Args:
        A: Left operand matrix with dimensions (m × k).
        B: Right operand matrix with dimensions (k × n).

    Returns:
        The product matrix C with dimensions (m × n).

    Raises:
        MatrixStructureError: if either matrix is ragged.
        MatrixDimensionError: if inner dimensions do not match.
    """
    # Validate structure and obtain dimensions
    m, k_a = _validate_matrix(A)
    k_b, n = _validate_matrix(B)

    # Compatibility check
    if k_a != k_b:
        raise MatrixDimensionError(
            f"Incompatible dimensions: A is {m}×{k_a}, B is {k_b}×{n}."
        )

    # Fast path for empty result
    if m == 0 or n == 0:
        return [[] for _ in range(m)]

    # Pre‑compute B transposed to improve cache locality
    # B_T will be n × k
    B_T: List[List[Number]] = [[B[row][col] for row in range(k_a)] for col in range(n)]

    # Compute result
    result: Matrix = []
    for i in range(m):
        row_result: List[Number] = []
        a_row = A[i]
        for j in range(n):
            b_col = B_T[j]
            # Dot product of a_row and b_col
            dot = sum(a_row[t] * b_col[t] for t in range(k_a))
            row_result.append(dot)
        result.append(row_result)

    return result
