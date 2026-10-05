from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Iterable
import math


def _validate_vector(vec: Sequence[float], expected_len: int | None = None) -> List[float]:
    """Return a copy of vec as list and validate its length."""
    lst = list(vec)
    if expected_len is not None and len(lst) != expected_len:
        raise ValueError(f"Vector length {len(lst)} does not match expected {expected_len}")
    return lst


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length for dot product")
    return sum(x * y for x, y in zip(a, b))


def l2_norm(v: Sequence[float]) -> float:
    """Euclidean norm."""
    return math.sqrt(dot(v, v))


@dataclass
class GradientStep:
    """Container for a single gradient step."""
    index: int
    gradient: List[float] = field(repr=False)

    def __post_init__(self) -> None:
        self.gradient = _validate_vector(self.gradient)


class PathIntegralSurrogate:
    """
    Compute a path‑integral surrogate for multi‑step gradient inversion.

    For a sequence of gradients g_0, …, g_{T‑1} and a constant learning rate η,
    the surrogate S is defined as

        S = η * Σ_{t=0}^{T‑1} w_t * g_t

    where w_t = 1 - t / T (linear decay).  The surrogate approximates the
    cumulative effect of the updates and can be used as a proxy for the
    original data gradient.
    """

    def __init__(self, learning_rate: float, steps: int) -> None:
        if steps <= 0:
            raise ValueError("steps must be a positive integer")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.lr: float = learning_rate
        self.T: int = steps
        self._steps: List[GradientStep] = []
        self._dim: int | None = None

    def add_step(self, gradient: Sequence[float]) -> None:
        """Append a new gradient to the internal buffer."""
        if len(self._steps) >= self.T:
            raise RuntimeError("All steps already recorded")
        grad = _validate_vector(gradient, self._dim)
        if self._dim is None:
            self._dim = len(grad)
        self._steps.append(GradientStep(index=len(self._steps), gradient=grad))

    def compute_surrogate(self) -> List[float]:
        """Return the weighted surrogate vector."""
        if len(self._steps) != self.T:
            raise RuntimeError(f"Expected {self.T} steps, got {len(self._steps)}")
        surrogate = [0.0] * self._dim  # type: ignore[arg-type]
        for step in self._steps:
            weight = 1.0 - step.index / self.T
            for i, val in enumerate(step.gradient):
                surrogate[i] += self.lr * weight * val
        return surrogate

    def invert(self, surrogate: Sequence[float]) -> List[float]:
        """
        Produce a naive inversion of the surrogate back to an approximate
        original gradient.  The inverse operation rescales by the average
        weight used during construction.
        """
        if self._dim is None:
            raise RuntimeError("No gradient dimensions known")
        s = _validate_vector(surrogate, self._dim)
        avg_weight = sum(1.0 - t / self.T for t in range(self.T)) / self.T
        if avg_weight == 0:
            raise ZeroDivisionError("Average weight is zero")
        scale = 1.0 / (self.lr * avg_weight)
        return [val * scale for val in s]

    def reset(self) -> None:
        """Clear stored steps for reuse."""
        self._steps.clear()
        self._dim = None
