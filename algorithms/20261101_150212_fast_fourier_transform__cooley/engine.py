from __future__ import annotations
import math
from typing import List, Sequence, Tuple
from .types import Signal, FFTResult

def _is_power_of_two(n: int) -> bool:
    """Return True if n is a power of two and n > 0."""
    return n > 0 and (n & (n - 1)) == 0

def _bit_reverse(value: int, bits: int) -> int:
    """Reverse the lowest `bits` bits of integer `value`."""
    rev = 0
    for _ in range(bits):
        rev = (rev << 1) | (value & 1)
        value >>= 1
    return rev

def _bit_reversed_permutation(data: List[complex]) -> List[complex]:
    """Return a new list where elements are reordered according to bit‑reversed indices."""
    n = len(data)
    bits = n.bit_length() - 1  # because n is power of two
    result = [0j] * n
    for i in range(n):
        rev = _bit_reverse(i, bits)
        result[rev] = data[i]
    return result

def fft(signal: Sequence[complex]) -> FFTResult:
    """
    Compute the forward Discrete Fourier Transform using the Cooley‑Tukey
    radix‑2 decimation‑in‑time algorithm.

    Parameters
    ----------
    signal : Sequence[complex]
        Input sequence. Length must be a power of two.

    Returns
    -------
    FFTResult
        List of complex frequency coefficients.
    """
    n = len(signal)
    if not _is_power_of_two(n):
        raise ValueError("Length of input must be a power of two.")
    # Work on a mutable copy
    a = _bit_reversed_permutation(list(signal))

    # Iterative Danielson‑Lanczos portion
    m = 2
    while m <= n:
        theta = -2.0 * math.pi / m
        w_m = complex(math.cos(theta), math.sin(theta))
        for k in range(0, n, m):
            w = 1.0 + 0j
            for j in range(m // 2):
                t = w * a[k + j + m // 2]
                u = a[k + j]
                a[k + j] = u + t
                a[k + j + m // 2] = u - t
                w *= w_m
        m <<= 1
    return FFTResult(a)

def ifft(spectrum: Sequence[complex]) -> Signal:
    """
    Compute the inverse DFT using the forward FFT algorithm and scaling.

    Parameters
    ----------
    spectrum : Sequence[complex]
        Frequency domain data (output of `fft`). Length must be a power of two.

    Returns
    -------
    Signal
        Time‑domain sequence (complex values).
    """
    n = len(spectrum)
    if not _is_power_of_two(n):
        raise ValueError("Length of input must be a power of two.")
    # Conjugate, forward FFT, conjugate again, then scale by 1/n
    conjugated = [x.conjugate() for x in spectrum]
    transformed = fft(conjugated)
    result = [x.conjugate() / n for x in transformed]
    return Signal(result)
