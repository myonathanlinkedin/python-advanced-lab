from __future__ import annotations
from typing import List, Sequence
import math
import cmath

from .types import FFTResult, FFTInputError

def _is_power_of_two(n: int) -> bool:
    """Return True if n is a power of two (and > 0)."""
    return n > 0 and (n & (n - 1)) == 0

def _naive_dft(signal: Sequence[complex]) -> FFTResult:
    """Compute the discrete Fourier transform using the definition (O(N^2))."""
    N = len(signal)
    result: List[complex] = []
    for k in range(N):
        s = 0j
        for n, x_n in enumerate(signal):
            angle = -2 * math.pi * k * n / N
            s += x_n * cmath.exp(1j * angle)
        result.append(s)
    return result

def fft(signal: Sequence[complex]) -> FFTResult:
    """
    Compute the Cooley–Tukey radix-2 FFT of a 1‑D signal.
    
    Parameters
    ----------
    signal : Sequence[complex]
        Input sequence. Length must be a power of two.
    
    Returns
    -------
    List[complex]
        The transformed sequence.
    
    Raises
    ------
    FFTInputError
        If the length of ``signal`` is not a power of two.
    """
    N = len(signal)
    if not _is_power_of_two(N):
        raise FFTInputError(f"Signal length {N} is not a power of two.")
    # Base case
    if N == 1:
        return [signal[0]]
    # Split even and odd indexed elements
    even = fft(signal[0::2])
    odd = fft(signal[1::2])
    # Combine
    result: List[complex] = [0j] * N
    for k in range(N // 2):
        twiddle = cmath.exp(-2j * cmath.pi * k / N) * odd[k]
        result[k] = even[k] + twiddle
        result[k + N // 2] = even[k] - twiddle
    return result

def ifft(spectrum: Sequence[complex]) -> FFTResult:
    """
    Compute the inverse FFT using the forward FFT algorithm.
    
    Parameters
    ----------
    spectrum : Sequence[complex]
        Frequency‑domain representation. Length must be a power of two.
    
    Returns
    -------
    List[complex]
        Time‑domain signal (scaled by 1/N).
    
    Raises
    ------
    FFTInputError
        If the length of ``spectrum`` is not a power of two.
    """
    N = len(spectrum)
    if not _is_power_of_two(N):
        raise FFTInputError(f"Spectrum length {N} is not a power of two.")
    # Conjugate, forward FFT, conjugate again, then scale
    conjugated = [x.conjugate() for x in spectrum]
    transformed = fft(conjugated)
    return [x.conjugate() / N for x in transformed]
