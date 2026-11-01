from __future__ import annotations
from typing import List, Sequence, Tuple, NewType

# Alias for a sequence of complex numbers representing a signal
Signal = NewType('Signal', List[complex])

# Result of an FFT operation (complex frequency bins)
FFTResult = NewType('FFTResult', List[complex])
