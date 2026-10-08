from __future__ import annotations

from typing import List, Sequence, Tuple, TypeVar

# Generic numeric type for matrix entries
Number = TypeVar("Number", int, float, complex)

# A matrix is a list of rows, each row is a list of numbers.
Matrix = List[List[Number]]


class MatrixDimensionError(ValueError):
    """Raised when matrix dimensions are incompatible for an operation."""
    pass


class MatrixStructureError(ValueError):
    """Raised when a matrix is not well‑formed (e.g., ragged rows)."""
    pass


def _validate_matrix(mat: Matrix) -> Tuple[int, int]:
    """
    Validate that ``mat`` is a well‑formed rectangular matrix.

    Returns:
        (row_count, col_count)

    Raises:
        MatrixStructureError: if rows have differing lengths.
    """
    if not isinstance(mat, list):
        raise MatrixStructureError("Matrix must be a list of rows.")
    if len(mat) == 0:
        return 0, 0
    col_count = None
    for idx, row in enumerate(mat):
        if not isinstance(row, list):
            raise MatrixStructureError(f"Row {idx} is not a list.")
        if col_count is None:
            col_count = len(row)
        elif len(row) != col_count:
            raise MatrixStructureError(
                f"Row {idx} length {len(row)} differs from previous length {col_count}."
            )
    return len(mat), col_count
