import cmath
from typing import List, Sequence

__all__ = ["fft", "ifft", "convolve", "is_power_of_two", "next_power_of_two"]


def is_power_of_two(n: int) -> bool:
    """Return True if n is a power of two (n > 0)."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n (n > 0)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if is_power_of_two(n):
        return n
    p = 1
    while p < n:
        p <<= 1
    return p


def _bit_reverse_copy(x: Sequence[complex]) -> List[complex]:
    """Reorder input according to bit-reversed indices."""
    n = len(x)
    result = [0j] * n
    j = 0
    for i in range(n):
        result[j] = x[i]
        m = n >> 1
        while j & m:
            j ^= m
            m >>= 1
        j ^= m
    return result


def fft(x: Sequence[complex]) -> List[complex]:
    """
    Compute the Radix-2 Cooley‑Tukey FFT of a sequence whose length is a power of two.
    Returns a new list containing the transformed values.
    """
    n = len(x)
    if n == 0:
        return []
    if not is_power_of_two(n):
        raise ValueError("Length of input must be a power of two")
    a = _bit_reverse_copy(x)

    length = 2
    while length <= n:
        half = length // 2
        theta = -2j * cmath.pi / length
        wlen = cmath.exp(theta)
        for i in range(0, n, length):
            w = 1 + 0j
            for k in range(half):
                u = a[i + k]
                v = a[i + k + half] * w
                a[i + k] = u + v
                a[i + k + half] = u - v
                w *= wlen
        length <<= 1
    return a


def ifft(x: Sequence[complex]) -> List[complex]:
    """
    Compute the inverse FFT using the conjugate method.
    Returns a new list containing the inverse transformed values.
    """
    n = len(x)
    if n == 0:
        return []
    if not is_power_of_two(n):
        raise ValueError("Length of input must be a power of two")
    # Conjugate, forward FFT, conjugate again, scale by 1/n
    conj = [xi.conjugate() for xi in x]
    y = fft(conj)
    return [yi.conjugate() / n for yi in y]


def convolve(a: Sequence[complex], b: Sequence[complex]) -> List[complex]:
    """
    Compute the linear convolution of two sequences using FFT.
    The result length is len(a) + len(b) - 1.
    """
    if not a or not b:
        return []
    m = len(a) + len(b) - 1
    n = next_power_of_two(m)
    # Pad both sequences
    a_pad = list(a) + [0j] * (n - len(a))
    b_pad = list(b) + [0j] * (n - len(b))
    # FFT, pointwise multiply, inverse FFT
    A = fft(a_pad)
    B = fft(b_pad)
    C = [A[i] * B[i] for i in range(n)]
    c_full = ifft(C)
    # Truncate to the exact convolution length
    return c_full[:m]
