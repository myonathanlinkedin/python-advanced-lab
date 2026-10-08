from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


Vector = List[float]


@dataclass(frozen=True)
class CSRMatrix:
    """
    Compressed Sparse Row (CSR) representation of a sparse matrix.

    Attributes
    ----------
    data : List[float]
        Non‑zero values of the matrix.
    indices : List[int]
        Column indices corresponding to each value in ``data``.
    indptr : List[int]
        Index pointers for row start in ``data``/``indices``.
        Length is ``n_rows + 1``; ``indptr[i]`` is the start index of row ``i``.
    shape : Tuple[int, int]
        Matrix dimensions as ``(n_rows, n_cols)``.
    """
    data: List[float]
    indices: List[int]
    indptr: List[int]
    shape: Tuple[int, int]

    def __post_init__(self) -> None:
        n_rows, n_cols = self.shape
        if len(self.indptr) != n_rows + 1:
            raise ValueError("indptr length must be n_rows + 1")
        if len(self.data) != len(self.indices):
            raise ValueError("data and indices must have the same length")
        if self.indptr[-1] != len(self.data):
            raise ValueError("indptr last element must equal number of non‑zero entries")
        # Validate column indices are within bounds
        for col in self.indices:
            if not (0 <= col < n_cols):
                raise ValueError(f"column index {col} out of bounds for matrix with {n_cols} columns")
