from __future__ import annotations
from typing import List
from .types import CSRMatrix, Vector


def spmv(matrix: CSRMatrix, vector: Vector) -> Vector:
    """
    Perform sparse matrix‑vector multiplication (y = A * x).

    Parameters
    ----------
    matrix : CSRMatrix
        Sparse matrix in CSR format.
    vector : Vector
        Dense vector with length equal to ``matrix.shape[1]``.

    Returns
    -------
    Vector
        Resulting dense vector of length ``matrix.shape[0]``.

    Raises
    ------
    ValueError
        If the dimensions of ``matrix`` and ``vector`` are incompatible.
    """
    n_rows, n_cols = matrix.shape
    if len(vector) != n_cols:
        raise ValueError(
            f"Dimension mismatch: matrix has {n_cols} columns but vector length is {len(vector)}"
        )

    result: List[float] = [0.0 for _ in range(n_rows)]

    for row in range(n_rows):
        row_start = matrix.indptr[row]
        row_end = matrix.indptr[row + 1]
        row_sum = 0.0
        for idx in range(row_start, row_end):
            col = matrix.indices[idx]
            val = matrix.data[idx]
            row_sum += val * vector[col]
        result[row] = row_sum

    return result
