from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

# A matrix is represented as a list of rows, each row is a list of floats.
Matrix = List[List[float]]
Vector = List[float]

@dataclass(frozen=True)
class SVDResult:
    """Container for the result of a singular value decomposition."""
    u: Matrix          # Left singular vectors (m x r)
    s: Vector          # Singular values (length r, sorted descending)
    vt: Matrix         # Right singular vectors transposed (r x n)
