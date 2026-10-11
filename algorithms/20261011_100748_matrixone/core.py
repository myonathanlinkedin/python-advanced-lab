from __future__ import annotations
from typing import Sequence, Tuple, List, Union, overload
import math

Number = Union[int, float]


class Matrix:
    """
    Immutable matrix class supporting basic linear algebra operations.
    Internal representation is a tuple of tuples for hashability and safety.
    """

    __slots__ = ("_data", "_rows", "_cols")

    def __init__(self, data: Sequence[Sequence[Number]]) -> None:
        if not data:
            raise ValueError("Matrix cannot be empty")
        rows = len(data)
        cols = len(data[0])
        if any(len(row) != cols for row in data):
            raise ValueError("All rows must have the same length")
        # Convert to tuple of tuples for immutability
        self._data: Tuple[Tuple[Number, ...], ...] = tuple(tuple(float(x) for x in row) for row in data)
        self._rows = rows
        self._cols = cols

    @property
    def shape(self) -> Tuple[int, int]:
        return self._rows, self._cols

    def __getitem__(self, idx: int) -> Tuple[Number, ...]:
        return self._data[idx]

    def __repr__(self) -> str:
        return f"Matrix({[list(row) for row in self._data]})"

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------
    @staticmethod
    def zeros(rows: int, cols: int) -> Matrix:
        if rows <= 0 or cols <= 0:
            raise ValueError("Dimensions must be positive")
        return Matrix([[0.0] * cols for _ in range(rows)])

    @staticmethod
    def ones(rows: int, cols: int) -> Matrix:
        if rows <= 0 or cols <= 0:
            raise ValueError("Dimensions must be positive")
        return Matrix([[1.0] * cols for _ in range(rows)])

    @staticmethod
    def identity(size: int) -> Matrix:
        if size <= 0:
            raise ValueError("Size must be positive")
        return Matrix([[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)])

    @staticmethod
    def from_list(data: List[List[Number]]) -> Matrix:
        return Matrix(data)

    # -------------------------------------------------------------------------
    # Elementwise operations
    # -------------------------------------------------------------------------
    def _check_same_shape(self, other: Matrix) -> None:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

    def __add__(self, other: Matrix) -> Matrix:
        self._check_same_shape(other)
        rows, cols = self.shape
        result = [
            [self._data[i][j] + other._data[i][j] for j in range(cols)]
            for i in range(rows)
        ]
        return Matrix(result)

    def __sub__(self, other: Matrix) -> Matrix:
        self._check_same_shape(other)
        rows, cols = self.shape
        result = [
            [self._data[i][j] - other._data[i][j] for j in range(cols)]
            for i in range(rows)
        ]
        return Matrix(result)

    # -------------------------------------------------------------------------
    # Scalar and matrix multiplication
    # -------------------------------------------------------------------------
    @overload
    def __mul__(self, other: Number) -> Matrix: ...

    @overload
    def __mul__(self, other: Matrix) -> Matrix: ...

    def __mul__(self, other: Union[Number, Matrix]) -> Matrix:
        if isinstance(other, (int, float)):
            rows, cols = self.shape
            result = [
                [self._data[i][j] * float(other) for j in range(cols)]
                for i in range(rows)
            ]
            return Matrix(result)
        elif isinstance(other, Matrix):
            if self._cols != other._rows:
                raise ValueError(
                    f"Incompatible dimensions for multiplication: {self.shape} and {other.shape}"
                )
            return self._matmul(other)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Matrix' and '{type(other)}'")

    def __rmul__(self, other: Number) -> Matrix:
        return self * other

    def _matmul(self, other: Matrix) -> Matrix:
        """Classic O(n³) matrix multiplication."""
        m, n = self.shape
        _, p = other.shape
        # Pre‑compute transpose of other for cache‑friendliness
        other_t = other.transpose()
        result = []
        for i in range(m):
            row = []
            for j in range(p):
                s = 0.0
                for k in range(n):
                    s += self._data[i][k] * other_t._data[j][k]
                row.append(s)
            result.append(row)
        return Matrix(result)

    # -------------------------------------------------------------------------
    # Transpose
    # -------------------------------------------------------------------------
    def transpose(self) -> Matrix:
        rows, cols = self.shape
        transposed = [
            [self._data[i][j] for i in range(rows)] for j in range(cols)
        ]
        return Matrix(transposed)

    # -------------------------------------------------------------------------
    # Determinant (LU decomposition with partial pivoting)
    # -------------------------------------------------------------------------
    def determinant(self) -> float:
        if self._rows != self._cols:
            raise ValueError("Determinant is defined only for square matrices")
        n = self._rows
        # Copy data to mutable list of lists
        A = [list(row) for row in self._data]
        perm_sign = 1.0
        for k in range(n):
            # Partial pivot
            max_row = max(range(k, n), key=lambda i: abs(A[i][k]))
            if math.isclose(A[max_row][k], 0.0, abs_tol=1e-12):
                return 0.0
            if max_row != k:
                A[k], A[max_row] = A[max_row], A[k]
                perm_sign *= -1.0
            pivot = A[k][k]
            for i in range(k + 1, n):
                factor = A[i][k] / pivot
                A[i][k] = factor
                for j in range(k + 1, n):
                    A[i][j] -= factor * A[k][j]
        det = perm_sign
        for i in range(n):
            det *= A[i][i]
        return det

    # -------------------------------------------------------------------------
    # Inverse (Gauss‑Jordan elimination)
    # -------------------------------------------------------------------------
    def inverse(self) -> Matrix:
        if self._rows != self._cols:
            raise ValueError("Inverse is defined only for square matrices")
        n = self._rows
        # Augment with identity
        A = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(self._data)]
        # Forward elimination
        for k in range(n):
            # Pivot
            max_row = max(range(k, n), key=lambda i: abs(A[i][k]))
            if math.isclose(A[max_row][k], 0.0, abs_tol=1e-12):
                raise ValueError("Matrix is singular and cannot be inverted")
            if max_row != k:
                A[k], A[max_row] = A[max_row], A[k]
            pivot = A[k][k]
            # Normalize pivot row
            A[k] = [x / pivot for x in A[k]]
            # Eliminate other rows
            for i in range(n):
                if i == k:
                    continue
                factor = A[i][k]
                A[i] = [a_i - factor * a_k for a_i, a_k in zip(A[i], A[k])]
        # Extract inverse
        inv_data = [row[n:] for row in A]
        return Matrix(inv_data)
