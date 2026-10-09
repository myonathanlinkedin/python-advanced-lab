from __future__ import annotations
from typing import List, Sequence, Tuple, Callable, Any
from dataclasses import dataclass

# Basic linear algebra type aliases
Vector = List[float]
Matrix = List[List[float]]

@dataclass(frozen=True)
class LinearSystem:
    """Immutable container for a symmetric positive‑definite linear system Ax = b."""
    A: Matrix
    b: Vector

    def __post_init__(self) -> None:
        # Basic validation – dimensions must match and A must be square
        n = len(self.A)
        if any(len(row) != n for row in self.A):
            raise ValueError("Matrix A must be square.")
        if len(self.b) != n:
            raise ValueError("Vector b length must match dimensions of A.")
        # No SPD check here – the solver will assume it.
