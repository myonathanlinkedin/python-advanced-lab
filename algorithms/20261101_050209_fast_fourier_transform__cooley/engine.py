import cmath
import math
from typing import List

from custom_types import Signal, FFTResult


def _is_power_of_two(n: int) -> bool:
    """Return True iff n is a power of two (including 0 → False)."""
    return n > 0 and (n & (n - 1)) == 0


def _bit_reverse(value: int, bits: int) -> int:
    """Reverse the lowest `bits` bits of `value`."""
    rev = 0
    for _ in range(bits):
        rev = (rev << 1) | (value & 1)
        value >>= 1
    return rev


def _bit_reverse_copy(x: List[complex]) -> List[complex]:
    """Return a copy of `x` reordered by bit‑reversed indices."""
    n = len(x)
    bits = n.bit_length() - 1
    result = [0j] * n
    for i in range(n):
        rev = _bit_reverse(i, bits)
        result[rev] = x[i]
    return result


def fft(signal: Signal) -> FFTResult:
    """
    Compute the radix‑2 Cooley‑Tukey FFT of `signal`.

    Preconditions:
        - len(signal.samples) is a power of two (zero length returns empty list).

    Returns:
        FFTResult containing the transformed sequence.
    """
    x = signal.samples
    n = len(x)

    if n == 0:
        return FFTResult([])

    if not _is_power_of_two(n):
        raise ValueError("FFT length must be a power of two")

    # Bit‑reversed copy
    X = _bit_reverse_copy(x)

    # Iterative Danielson‑Lanczos portion
    m = 2
    while m <= n:
        # Twiddle factor for this stage
        w_m = cmath.exp(-2j * math.pi / m)
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

    return FFTResult(X)


def ifft(signal: Signal) -> FFTResult:
    """
    Compute the inverse FFT using the conjugate‑symmetry property.

    Returns:
        FFTResult containing the time‑domain sequence (scaled by 1/N).
    """
    n = len(signal.samples)

    if n == 0:
        return FFTResult([])

    if not _is_power_of_two(n):
        raise ValueError("IFFT length must be a power of two")

    # Conjugate, forward FFT, conjugate again, then scale
    conjugated = [c.conjugate() for c in signal.samples]
    transformed = fft(Signal(conjugated)).transformed
    inv = [c.conjugate() / n for c in transformed]
    return FFTResult(inv)
