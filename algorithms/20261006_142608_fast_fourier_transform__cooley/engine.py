from __future__ import annotations

import math
from typing import List, Sequence

from .types import FFTResult, Signal

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()

def _pad_signal(signal: Sequence[complex]) -> Signal:
    """Pad the signal with zeros to the next power-of-two length."""
    target_len = _next_power_of_two(len(signal))
    return list(signal) + [0j] * (target_len - len(signal))

def _fft_recursive(x: Signal) -> Signal:
    """Recursive Cooley–Tukey radix‑2 FFT (in‑place not required)."""
    n = len(x)
    if n == 1:
        return x.copy()
    if not _is_power_of_two(n):
        raise ValueError("Length of input must be a power of two for recursive FFT.")
    even = _fft_recursive(x[0::2])
    odd = _fft_recursive(x[1::2])
    combined: Signal = [0j] * n
    for k in range(n // 2):
        twiddle = math.e**(-2j * math.pi * k / n)
        combined[k] = even[k] + twiddle * odd[k]
        combined[k + n // 2] = even[k] - twiddle * odd[k]
    return combined

def _ifft_recursive(x: Signal) -> Signal:
    """Recursive inverse FFT using conjugation symmetry."""
    n = len(x)
    if n == 1:
        return x.copy()
    if not _is_power_of_two(n):
        raise ValueError("Length of input must be a power of two for recursive IFFT.")
    even = _ifft_recursive(x[0::2])
    odd = _ifft_recursive(x[1::2])
    combined: Signal = [0j] * n
    for k in range(n // 2):
        twiddle = math.e**(2j * math.pi * k / n)  # note sign reversal
        combined[k] = even[k] + twiddle * odd[k]
        combined[k + n // 2] = even[k] - twiddle * odd[k]
    return combined

def fft(signal: Sequence[complex]) -> FFTResult:
    """
    Compute the forward FFT of *signal*.
    The input is padded to the next power of two; the original length is stored.
    """
    padded = _pad_signal(signal)
    transformed = _fft_recursive(padded)
    return FFTResult(transformed=transformed, original_length=len(signal))

def ifft(transformed: Sequence[complex]) -> FFTResult:
    """
    Compute the inverse FFT.
    The input is padded to the next power of two; the original length is stored.
    The result is scaled by 1/N to obtain the original time‑domain values.
    """
    padded = _pad_signal(transformed)
    n = len(padded)
    raw = _ifft_recursive(padded)
    # Scale by 1/N
    scaled = [value / n for value in raw]
    return FFTResult(transformed=scaled, original_length=len(transformed))
