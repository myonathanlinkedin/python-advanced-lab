import cmath
from typing import List
from .types import Signal

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def _bit_reverse_copy(x: Signal) -> Signal:
    n = len(x)
    result: Signal = [0j] * n
    bits = n.bit_length() - 1
    for i in range(n):
        rev = 0
        for j in range(bits):
            if (i >> j) & 1:
                rev |= 1 << (bits - 1 - j)
        result[rev] = x[i]
    return result

def fft(x: Signal) -> Signal:
    """
    Cooley‑Tukey radix‑2 decimation‑in‑time FFT.
    Input length must be a power of two.
    Returns a new list containing the frequency domain representation.
    """
    n = len(x)
    if n == 0:
        raise ValueError("Input signal must contain at least one element")
    if not _is_power_of_two(n):
        raise ValueError("Length of input must be a power of two")
    X = _bit_reverse_copy(x)
    m = 2
    while m <= n:
        theta = -2j * cmath.pi / m
        w_m = cmath.exp(theta)
        for k in range(0, n, m):
            w = 1 + 0j
            half = m // 2
            for j in range(half):
                t = w * X[k + j + half]
                u = X[k + j]
                X[k + j] = u + t
                X[k + j + half] = u - t
                w *= w_m
        m <<= 1
    return X

def ifft(X: Signal) -> Signal:
    """
    Inverse FFT using conjugation method.
    Returns the time‑domain signal.
    """
    n = len(X)
    if n == 0:
        raise ValueError("Input signal must contain at least one element")
    if not _is_power_of_two(n):
        raise ValueError("Length of input must be a power of two")
    # Conjugate, forward FFT, conjugate again, then scale
    conj = [x.conjugate() for x in X]
    y = fft(conj)
    return [ (val.conjugate() / n) for val in y ]

def naive_dft(x: Signal) -> Signal:
    """
    Direct O(N^2) DFT for verification purposes.
    """
    n = len(x)
    result: Signal = []
    for k in range(n):
        s = 0j
        for t in range(n):
            angle = -2j * cmath.pi * t * k / n
            s += x[t] * cmath.exp(angle)
        result.append(s)
    return result
