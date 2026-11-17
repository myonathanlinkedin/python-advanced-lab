import math
from typing import List, Sequence

__all__ = ["fft", "ifft", "_naive_dft"]

def _is_power_of_two(n: int) -> bool:
    """Return True if n is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0

def _bit_reverse(n: int, bits: int) -> int:
    """Return the bit-reversed value of n with given number of bits."""
    rev = 0
    for _ in range(bits):
        rev = (rev << 1) | (n & 1)
        n >>= 1
    return rev

def _bit_reverse_copy(a: List[complex]) -> List[complex]:
    """Return a new list with elements of a reordered by bit-reversal."""
    n = len(a)
    bits = n.bit_length() - 1
    result = [0] * n
    for i in range(n):
        j = _bit_reverse(i, bits)
        result[j] = a[i]
    return result

def fft(a: Sequence[complex]) -> List[complex]:
    """
    Compute the radix-2 Cooley–Tukey FFT of the input sequence a.
    The length of a must be a positive power of two.
    Returns a new list containing the DFT of a.
    """
    n = len(a)
    if not _is_power_of_two(n):
        raise ValueError("Length of input must be a power of two")
    # Convert to list of complex numbers
    x = [complex(val) for val in a]
    # Bit-reversal permutation
    x = _bit_reverse_copy(x)
    # Iterative FFT
    m = 2
    while m <= n:
        half_m = m // 2
        theta = -2 * math.pi / m
        w_m = complex(math.cos(theta), math.sin(theta))
        for k in range(0, n, m):
            w = 1 + 0j
            for j in range(half_m):
                t = w * x[k + j + half_m]
                u = x[k + j]
                x[k + j] = u + t
                x[k + j + half_m] = u - t
                w *= w_m
        m <<= 1
    return x

def ifft(a: Sequence[complex]) -> List[complex]:
    """
    Compute the inverse FFT of the input sequence a.
    The length of a must be a positive power of two.
    Returns a new list containing the inverse DFT of a.
    """
    n = len(a)
    if not _is_power_of_two(n):
        raise ValueError("Length of input must be a power of two")
    # Conjugate, FFT, conjugate, scale
    conj = [complex(val).conjugate() for val in a]
    y = fft(conj)
    return [x.conjugate() / n for x in y]

def _naive_dft(a: Sequence[complex]) -> List[complex]:
    """
    Naive O(n^2) DFT implementation for verification purposes.
    """
    n = len(a)
    result = []
    for k in range(n):
        s = 0+0j
        for n_idx, val in enumerate(a):
            angle = -2 * math.pi * k * n_idx / n
            s += val * complex(math.cos(angle), math.sin(angle))
        result.append(s)
    return result
