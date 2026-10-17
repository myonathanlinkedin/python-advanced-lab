from typing import List, Sequence, Tuple

# Type alias for a sequence of complex numbers representing a signal
Signal = Sequence[complex]

# Result of FFT is a list of complex numbers
FFTResult = List[complex]

# Custom exception for FFT input validation
class FFTInputError(ValueError):
    """Raised when the input signal does not satisfy FFT requirements."""
    pass
