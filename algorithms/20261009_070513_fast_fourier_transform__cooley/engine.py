from __future__ import annotations

import math
import cmath
from typing import List, Sequence

from .types import FFTResult, ComplexSequence

def _is_power_of_two(n: int) -> bool:
    """Return True if n is a power of two (n > 0)."""
    return n > 0 and (n & (n - 1)) == 0

def _bit_reverse_indices(n: int) -> List[int]:
    """Return a list of bit‑reversed indices for length n (n must be power of two)."""
    bits = n.bit_length() - 1
    rev = [0] * n
    for i in range(n):
        rev_i = 0
        for j in range(bits):
            if (i >> j) & 1:
                rev_i |= 1 << (bits - 1 - j)
        rev[i] = rev_i
    return rev

def fft(signal: ComplexSequence) -> FFTResult:
    """
    Compute the radix‑2 Cooley‑Tukey FFT of a 1‑dimensional complex signal.

    Parameters
    ----------
    signal: Sequence[complex]
        Input sequence. Length must be a power of two.

    Returns
    -------
    FFTResult
        Immutable container holding the transformed values.

    Raises
    ------
    ValueError
        If the length of ``signal`` is not a power of two.
    """
    n = len(signal)
    if not _is_power_of_two(n):
        raise ValueError("FFT length must be a power of two.")
    # Copy to mutable list of complex numbers
    a: List[complex] = list(signal)

    # Bit‑reversal permutation
    rev = _bit_reverse_indices(n)
    a = [a[rev[i]] for i in range(n)]

    # Iterative butterfly
    size = 2
    while size <= n:
        half = size // 2
        theta = -2.0 * math.pi / size
        w_m = cmath.exp(complex(0, theta))  # principal twiddle factor
        for k in range(0, n, size):
            w = 1+0j
            for j in range(half):
                t = w * a[k + j + half]
                u = a[k + j]
                a[k + j] = u + t
                a[k + j + half] = u - t
                w *= w_m
        size <<= 1

    return FFTResult(values=tuple(a))

def ifft(spectrum: ComplexSequence) -> FFTResult:
    """
    Compute the inverse FFT using the conjugate‑symmetry method.

    Parameters
    ----------
    spectrum: Sequence[complex]
        Frequency‑domain data (output of :func:`fft`). Length must be a power of two.

    Returns
    -------
    FFTResult
        Time‑domain signal (scaled by 1/N).
    """
    n = len(spectrum)
    if not _is_power_of_two(n):
        raise ValueError("IFFT length must be a power of two.")
    # Conjugate, forward FFT, conjugate again, then scale
    conjugated = [x.conjugate() for x in spectrum]
    transformed = fft(conjugated).values
    result = [x.conjugate() / n for x in transformed]
    return FFTResult(values=tuple(result))

def naive_dft(signal: ComplexSequence) -> List[complex]:
    """
    Compute the discrete Fourier transform directly (O(N²)).
    Used for verification in unit tests.
    """
    n = len(signal)
    result: List[complex] = []
    for k in range(n):
        s = 0+0j
        for t, x in enumerate(signal):
            angle = -2.0 * math.pi * t * k / n
            s += x * cmath.exp(complex(0, angle))
        result.append(s)
    return result
