from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional

# Basic linear algebra type aliases
Vector = List[float]
Matrix = List[List[float]]

@dataclass(frozen=True)
class LinearSystem:
    """Immutable container for a symmetric positive‑definite linear system Ax = b."""
    A: Matrix
    b: Vector

    def __post_init__(self) -> None:
        n = len(self.A)
        if n == 0:
            raise ValueError("Matrix A must be non‑empty.")
        if any(len(row) != n for row in self.A):
            raise ValueError("Matrix A must be square.")
        if len(self.b) != n:
            raise ValueError("Vector b length must match dimensions of A.")
        # Minimal sanity check for symmetry (exact for pure Python floats)
        for i in range(n):
            for j in range(i + 1, n):
                if self.A[i][j] != self.A[j][i]:
                    raise ValueError("Matrix A must be symmetric.")
        # Positive‑definiteness is not checked here (expensive); caller must ensure it.
