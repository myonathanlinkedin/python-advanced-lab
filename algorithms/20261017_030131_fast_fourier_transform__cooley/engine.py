from __future__ import annotations

import math
from typing import List, Tuple

from .types import ComplexSignal, FFTConfig

def _is_power_of_two(n: int) -> bool:
    """Return True if n is a power of two (including 0)."""
    return n != 0 and (n & (n - 1) == 0)

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

def _bit_reverse_copy(signal: ComplexSignal) -> ComplexSignal:
    """Reorder `signal` according to bit‑reversed indices."""
    n = len(signal)
    if n == 0:
        return []
    rev = _bit_reverse_indices(n)
    return [signal[rev[i]] for i in range(n)]

def _fft_iterative(signal: ComplexSignal, inverse: bool = False) -> ComplexSignal:
    """
    In‑place iterative Cooley‑Tukey radix‑2 FFT.
    Parameters:
        signal: Input sequence (length must be a power of two).
        inverse: If True, compute the inverse FFT.
    Returns:
        The transformed sequence as a new list.
    """
    n = len(signal)
    if n == 0:
        return []
    if not _is_power_of_two(n):
        raise ValueError("Length of input must be a power of two.")
    # Copy and bit‑reverse order
    a = _bit_reverse_copy(signal)

    # Main loops
    half_size = 1
    while half_size < n:
        phase_shift_step = (2 * math.pi / (2 * half_size)) * (-1 if not inverse else 1)
        w_phase_shift = complex(math.cos(phase_shift_step), math.sin(phase_shift_step))
        for k in range(0, n, 2 * half_size):
            w = 1+0j
            for j in range(half_size):
                t = w * a[k + j + half_size]
                u = a[k + j]
                a[k + j] = u + t
                a[k + j + half_size] = u - t
                w *= w_phase_shift
        half_size <<= 1

    # For inverse FFT, divide by n
    if inverse:
        a = [x / n for x in a]
    return a

def fft(signal: ComplexSignal) -> ComplexSignal:
    """
    Compute the forward FFT of `signal` using radix‑2 Cooley‑Tukey.
    The length of `signal` must be a power of two; otherwise a ValueError is raised.
    """
    return _fft_iterative(signal, inverse=False)

def ifft(signal: ComplexSignal) -> ComplexSignal:
    """
    Compute the inverse FFT of `signal`.
    The length of `signal` must be a power of two; otherwise a ValueError is raised.
    """
    return _fft_iterative(signal, inverse=True)

def dft_naive(signal: ComplexSignal) -> ComplexSignal:
    """
    Naïve O(N²) discrete Fourier transform for verification purposes.
    Works for any length (including non‑powers of two).
    """
    n = len(signal)
    if n == 0:
        return []
    result: ComplexSignal = []
    for k in range(n):
        s = 0+0j
        for t in range(n):
            angle = -2 * math.pi * k * t / n
            s += signal[t] * complex(math.cos(angle), math.sin(angle))
        result.append(s)
    return result

def idft_naive(signal: ComplexSignal) -> ComplexSignal:
    """
    Naïve inverse DFT (scaled by 1/N) for verification.
    """
    n = len(signal)
    if n == 0:
        return []
    result: ComplexSignal = []
    for k in range(n):
        s = 0+0j
        for t in range(n):
            angle = 2 * math.pi * k * t / n
            s += signal[t] * complex(math.cos(angle), math.sin(angle))
        result.append(s / n)
    return result
