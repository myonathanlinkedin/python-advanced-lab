"""
core.py
Implementation of the Cooley‑Tukey Radix‑2 Fast Fourier Transform (FFT) and its inverse.
All functions operate on Python's built‑in complex type and use only the standard library.
"""

from __future__ import annotations

import cmath
import math
from typing import List, Sequence

__all__: List[str] = [
    "is_power_of_two",
    "next_power_of_two",
    "bit_reverse_permutation",
    "fft",
    "ifft",
    "naive_dft",
]


def is_power_of_two(n: int) -> bool:
    """Return True iff *n* is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Smallest power of two >= *n* (for n > 0)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 << (n - 1).bit_length()


def bit_reverse_permutation(data: List[complex]) -> List[complex]:
    """
    Return a new list where the elements are reordered according to the bit‑reversed
    indices of their original positions. Length must be a power of two.
    """
    n = len(data)
    if not is_power_of_two(n):
        raise ValueError("Length of data must be a power of two")
    result = [0j] * n
    bits = n.bit_length() - 1
    for i, val in enumerate(data):
        rev = int('{:0{width}b}'.format(i, width=bits)[::-1], 2)
        result[rev] = val
    return result


def _fft_iterative(x: List[complex]) -> List[complex]:
    """
    In‑place iterative radix‑2 Cooley‑Tukey FFT.
    The input list *x* is assumed to have length a power of two.
    Returns a new list containing the transformed values.
    """
    n = len(x)
    a = bit_reverse_permutation(x)  # reorder input

    # Main loops
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


def fft(x: Sequence[complex]) -> List[complex]:
    """
    Compute the discrete Fourier transform of *x* using the radix‑2 Cooley‑Tukey algorithm.
    The length of *x* must be a power of two; otherwise a ValueError is raised.
    """
    data = list(x)
    if not is_power_of_two(len(data)):
        raise ValueError("FFT length must be a power of two")
    return _fft_iterative(data)


def ifft(X: Sequence[complex]) -> List[complex]:
    """
    Compute the inverse discrete Fourier transform of *X*.
    The algorithm reuses the forward FFT by conjugating the input,
    applying FFT, conjugating the result and scaling by 1/N.
    Length must be a power of two.
    """
    n = len(X)
    if not is_power_of_two(n):
        raise ValueError("IFFT length must be a power of two")
    conj = [x.conjugate() for x in X]
    transformed = _fft_iterative(conj)
    return [y.conjugate() / n for y in transformed]


def naive_dft(x: Sequence[complex]) -> List[complex]:
    """
    Compute the DFT directly in O(N²) time.
    Used for verification in unit tests.
    """
    n = len(x)
    result: List[complex] = []
    for k in range(n):
        s = 0j
        for j, val in enumerate(x):
            angle = -2j * cmath.pi * j * k / n
            s += val * cmath.exp(angle)
        result.append(s)
    return result
