from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Signal:
    """Immutable container for a discrete-time signal."""
    samples: List[complex]


@dataclass(frozen=True)
class FFTResult:
    """Immutable container for the result of an FFT operation."""
    transformed: List[complex]
