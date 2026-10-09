from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

ComplexSequence = Sequence[complex]

@dataclass(frozen=True)
class FFTResult:
    """Immutable container for FFT output."""
    values: Tuple[complex, ...]
