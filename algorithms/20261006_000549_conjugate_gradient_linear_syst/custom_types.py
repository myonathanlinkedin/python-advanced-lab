from __future__ import annotations
from typing import List, Iterable, Protocol, runtime_checkable, Any
import math

Vector = List[float]
Matrix = List[List[float]]


def dot(v1: Vector, v2: Vector) -> float:
    """Compute dot product of two vectors."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of same length for dot product.")
    return sum(a * b for a, b in zip(v1, v2))


def norm(v: Vector) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(dot(v, v))


def scalar_mul(s: float, v: Vector) -> Vector:
    """Multiply vector by scalar."""
    return [s * x for x in v]


def vec_add(v1: Vector, v2: Vector) -> Vector:
    """Add two vectors."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of same length for addition.")
    return [a + b for a, b in zip(v1, v2)]


def vec_sub(v1: Vector, v2: Vector) -> Vector:
    """Subtract v2 from v1."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of same length for subtraction.")
    return [a - b for a, b in zip(v1, v2)]


def mat_vec_mul(A: Matrix, v: Vector) -> Vector:
    """Multiply matrix A by vector v."""
    n_rows = len(A)
    if n_rows == 0:
        raise ValueError("Matrix has no rows.")
    n_cols = len(A[0])
    if len(v) != n_cols:
        raise ValueError("Matrix column count must match vector length.")
    result: Vector = []
    for row in A:
        if len(row) != n_cols:
            raise ValueError("Inconsistent row length in matrix.")
        result.append(dot(row, v))
    return result


def is_square_matrix(A: Matrix) -> bool:
    """Check if matrix is square."""
    return all(len(row) == len(A) for row in A)
