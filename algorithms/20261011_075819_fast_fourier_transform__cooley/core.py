"""
core.py
Implementation of the radix‑2 Cooley–Tukey Fast Fourier Transform (FFT) and its inverse.
All functions operate on Python's built‑in complex type and require the input length
to be a power of two.
"""

from __future__ import annotations

import cmath
from typing import List, Sequence


def _is_power_of_two(n: int) -> bool:
    """Return True iff *n* is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def fft(x: Sequence[complex]) -> List[complex]:
    """
    Compute the forward FFT of a sequence *x* using the radix‑2 Cooley–Tukey algorithm.

    Parameters
    ----------
    x : Sequence[complex]
        Input sequence. Length must be a power of two and non‑empty.

    Returns
    -------
    List[complex]
        The transformed sequence.

    Raises
    ------
    ValueError
        If the input length is zero or not a power of two.
    """
    n = len(x)
    if n == 0:
        raise ValueError("Input sequence must not be empty")
    if not _is_power_of_two(n):
        raise ValueError("Length of input must be a power of two")

    # Copy to mutable list
    a = list(x)

    # Bit‑reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # Iterative Cooley‑Tukey
    length = 2
    while length <= n:
        half = length // 2
        # Twiddle factor for this stage: exp(-2πi / length)
        theta = -2j * cmath.pi / length
        w_len = cmath.exp(theta)
        for start in range(0, n, length):
            w = 1 + 0j
            for k in range(half):
                u = a[start + k]
                v = a[start + k + half] * w
                a[start + k] = u + v
                a[start + k + half] = u - v
                w *= w_len
        length <<= 1

    return a


def ifft(X: Sequence[complex]) -> List[complex]:
    """
    Compute the inverse FFT of a sequence *X*.

    The implementation uses the conjugate‑symmetry property:
        ifft(X) = conj(fft(conj(X))) / n

    Parameters
    ----------
    X : Sequence[complex]
        Frequency‑domain sequence. Length must be a power of two and non‑empty.

    Returns
    -------
    List[complex]
        Time‑domain sequence.

    Raises
    ------
    ValueError
        If the input length is zero or not a power of two.
    """
    n = len(X)
    if n == 0:
        raise ValueError("Input sequence must not be empty")
    if not _is_power_of_two(n):
        raise ValueError("Length of input must be a power of two")

    # Conjugate, forward FFT, conjugate again, then scale by 1/n
    conj_input = [x.conjugate() for x in X]
    transformed = fft(conj_input)
    return [y.conjugate() / n for y in transformed]
