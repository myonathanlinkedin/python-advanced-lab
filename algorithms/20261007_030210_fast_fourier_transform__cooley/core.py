import math
import cmath
from typing import List

def _bit_reverse_copy(x: List[complex]) -> List[complex]:
    """Return a new list with elements of x reordered by bit-reversed indices."""
    n = len(x)
    y = [0] * n
    j = 0
    for i in range(n):
        y[j] = x[i]
        m = n >> 1
        while m and (j & m):
            j ^= m
            m >>= 1
        j ^= m
    return y

def _fft_iterative(x: List[complex]) -> List[complex]:
    """Iterative Cooley–Tukey radix‑2 FFT."""
    n = len(x)
    if n & (n - 1):
        raise ValueError("Length of input must be a power of 2")
    y = _bit_reverse_copy(x)
    m = 2
    while m <= n:
        half_m = m >> 1
        w_m = cmath.exp(-2j * math.pi / m)
        for k in range(0, n, m):
            w = 1 + 0j
            for j in range(half_m):
                t = w * y[k + j + half_m]
                u = y[k + j]
                y[k + j] = u + t
                y[k + j + half_m] = u - t
                w *= w_m
        m <<= 1
    return y

def fft(x: List[complex]) -> List[complex]:
    """Compute the discrete Fourier transform of a list of complex numbers."""
    return _fft_iterative(x)

def ifft(x: List[complex]) -> List[complex]:
    """Compute the inverse discrete Fourier transform."""
    n = len(x)
    conjugated = [xi.conjugate() for xi in x]
    y = _fft_iterative(conjugated)
    return [yi.conjugate() / n for yi in y]
