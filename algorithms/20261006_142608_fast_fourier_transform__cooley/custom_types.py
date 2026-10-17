from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

Signal = List[complex]

@dataclass(frozen=True)
class FFTResult:
    """Container for FFT output."""
    transformed: Signal
    original_length: int

    def trimmed(self) -> Signal:
        """Return the transformed signal trimmed to the original length."""
        return self.transformed[:self.original_length]
