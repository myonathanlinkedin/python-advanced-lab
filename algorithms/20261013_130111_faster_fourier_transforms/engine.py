from __future__ import annotations

import cmath
import math
from typing import List, Tuple

from .types import Signal, FFTLengthError

def _is_power_of_two(n: int) -> bool:
    """Return True if n is a power of two (including 0)."""
    return n != 0 and (n & (n - 1) == 0)

def _bit_reverse_copy(a: Signal) -> Signal:
    """Reorder the input list `a` by bit‑reversed indices."""
    n = len(a)
    result: Signal = [0j] * n
    bits = n.bit_length() - 1
    for k in range(n):
        rev = int('{:0{width}b}'.format(k, width=bits)[::-1], 2)
        result[rev] = a[k]
    return result

def fft(signal: Signal) -> Signal:
    """
    Compute the radix‑2 Cooley‑Tukey FFT of `signal`.

    Parameters
    ----------
    signal : Signal
        Input sequence whose length must be a power of two.

    Returns
    -------
    Signal
        The transformed sequence (in‑place algorithm returns a new list).

    Raises
    ------
    FFTLengthError
        If `len(signal)` is not a power of two.
    """
    n = len(signal)
    if n == 0:
        return []
    if not _is_power_of_two(n):
        raise FFTLengthError(f"FFT length must be a power of two, got {n}")

    # Bit‑reversed copy
    a = _bit_reverse_copy(signal)

    # Iterative Danielson‑Lanczos portion
    m = 2
    while m <= n:
        theta = -2j * cmath.pi / m
        wm = cmath.exp(theta)
        for k in range(0, n, m):
            w = 1 + 0j
            for j in range(m // 2):
                t = w * a[k + j + m // 2]
                u = a[k + j]
                a[k + j] = u + t
                a[k + j + m // 2] = u - t
                w *= wm
        m <<= 1
    return a

def ifft(signal: Signal) -> Signal:
    """
    Compute the inverse FFT of `signal`.

    The inverse transform is obtained by conjugating the input,
    performing a forward FFT, conjugating the result and scaling by 1/n.

    Parameters
    ----------
    signal : Signal
        Input frequency‑domain sequence (length must be a power of two).

    Returns
    -------
    Signal
        Time‑domain sequence.

    Raises
    ------
    FFTLengthError
        If `len(signal)` is not a power of two.
    """
    n = len(signal)
    if n == 0:
        return []
    if not _is_power_of_two(n):
        raise FFTLengthError(f"FFT length must be a power of two, got {n}")

    # Conjugate, forward FFT, conjugate again, then scale.
    conjugated = [x.conjugate() for x in signal]
    transformed = fft(conjugated)
    result = [x.conjugate() / n for x in transformed]
    return result

def convolution(a: Signal, b: Signal) -> Signal:
    """
    Compute the circular convolution of two equal‑length signals using FFT.

    Both inputs must have the same length, which must be a power of two.

    Parameters
    ----------
    a, b : Signal
        Input sequences.

    Returns
    -------
    Signal
        Circular convolution result.

    Raises
    ------
    FFTLengthError
        If the input lengths are mismatched or not a power of two.
    """
    if len(a) != len(b):
        raise FFTLengthError("Input signals must have the same length for convolution.")
    n = len(a)
    if n == 0:
        return []
    if not _is_power_of_two(n):
        raise FFTLengthError(f"FFT length must be a power of two, got {n}")

    A = fft(a)
    B = fft(b)
    C = [A[i] * B[i] for i in range(n)]
    return ifft(C)
