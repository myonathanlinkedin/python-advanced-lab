from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

# Type alias for a one‑dimensional signal of complex numbers.
Signal = List[complex]

class FFTError(Exception):
    """Base class for FFT related errors."""
    pass

class FFTLengthError(FFTError, ValueError):
    """Raised when the input length is not a power of two."""
    pass
