from __future__ import annotations

from typing import List, Sequence, Tuple, overload

Matrix = List[List[float]]


class RectangularMatrix:
    """Simple immutable rectangular matrix supporting multiplication."""

    __slots__ = ("_data", "_rows", "_cols")

    def __init__(self, data: Sequence[Sequence[float]]) -> None:
        if not data:
            raise ValueError("Matrix must have at least one row")
        row_lengths = {len(row) for row in data}
        if len(row_lengths) != 1:
            raise ValueError("All rows must have the same length")
        self._rows: int = len(data)
        self._cols: int = row_lengths.pop()
        # Store a deep copy as list of lists of floats
        self._data: Matrix = [[float(val) for val in row] for row in data]

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def data(self) -> Matrix:
        # Return a copy to preserve immutability
        return [row[:] for row in self._data]

    def __getitem__(self, idx: int) -> List[float]:
        return self._data[idx]

    def __repr__(self) -> str:
        return f"RectangularMatrix({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RectangularMatrix):
            return NotImplemented
        return self._data == other._data

    def __mul__(self, other: "RectangularMatrix") -> "RectangularMatrix":
        """Matrix multiplication using the shared‑leg (inner dimension) rule."""
        if self.cols != other.rows:
            raise ValueError(
                f"Incompatible dimensions for multiplication: "
                f"{self.rows}x{self.cols} cannot be multiplied by {other.rows}x{other.cols}"
            )
        result = multiply_matrices(self._data, other._data)
        return RectangularMatrix(result)


@overload
def multiply_matrices(a: Matrix, b: Matrix) -> Matrix: ...
@overload
def multiply_matrices(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix: ...


def multiply_matrices(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    """
    Multiply two rectangular matrices using the classic O(m·k·n) algorithm.

    Args:
        a: Left matrix of shape (m, k).
        b: Right matrix of shape (k, n).

    Returns:
        Product matrix of shape (m, n).

    Raises:
        ValueError: If inner dimensions do not match.
    """
    if not a or not b:
        raise ValueError("Input matrices must be non‑empty")
    m = len(a)
    k = len(a[0])
    if any(len(row) != k for row in a):
        raise ValueError("All rows of the left matrix must have the same length")
    if any(len(row) != len(b[0]) for row in b):
        raise ValueError("All rows of the right matrix must have the same length")
    if k != len(b):
        raise ValueError(
            f"Incompatible inner dimensions: left matrix columns ({k}) != right matrix rows ({len(b)})"
        )
    n = len(b[0])

    # Initialise result matrix with zeros
    result: Matrix = [[0.0 for _ in range(n)] for _ in range(m)]

    # Classic triple‑loop multiplication
    for i in range(m):
        a_i = a[i]
        res_i = result[i]
        for p in range(k):
            a_ip = a_i[p]
            b_p = b[p]
            for j in range(n):
                res_i[j] += a_ip * b_p[j]
    return result
