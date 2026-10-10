from __future__ import annotations
from typing import List, Sequence, Tuple, Callable

# Basic linear algebra type aliases
Vector = List[float]
Matrix = List[List[float]]

# Callable type for matrix-vector multiplication (useful for large or implicit matrices)
MatVec = Callable[[Vector], Vector]

# Result of the CG solver
class CGResult:
    """
    Container for the result of the Conjugate Gradient algorithm.

    Attributes
    ----------
    x : Vector
        Approximate solution vector.
    iterations : int
        Number of iterations performed.
    residual_norm : float
        Euclidean norm of the final residual (||b - A·x||_2).
    converged : bool
        True if the algorithm stopped because the residual norm fell below the tolerance.
    """
    __slots__ = ("x", "iterations", "residual_norm", "converged")

    def __init__(self, x: Vector, iterations: int, residual_norm: float, converged: bool) -> None:
        self.x = x
        self.iterations = iterations
        self.residual_norm = residual_norm
        self.converged = converged

    def __repr__(self) -> str:
        return (f"CGResult(x={self.x}, iterations={self.iterations}, "
                f"residual_norm={self.residual_norm:.2e}, converged={self.converged})")
