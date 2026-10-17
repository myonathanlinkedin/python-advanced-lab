from __future__ import annotations

from typing import List, Sequence, Tuple, Callable, Any

# Type alias for a sequence of complex numbers (signal)
ComplexSignal = List[complex]

# Type alias for a function that transforms a signal
SignalTransform = Callable[[ComplexSignal], ComplexSignal]

# Simple data class for FFT configuration (currently unused but extensible)
class FFTConfig:
    """
    Configuration holder for FFT operations.
    Attributes:
        inverse (bool): Whether to perform the inverse FFT.
    """
    def __init__(self, inverse: bool = False) -> None:
        self.inverse = inverse

    def __repr__(self) -> str:
        return f"FFTConfig(inverse={self.inverse})"
