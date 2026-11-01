import math
import cmath
from typing import List, Sequence


def dft(x: Sequence[complex]) -> List[complex]:
    """
    Compute the discrete Fourier transform (DFT) of a sequence.

    Args:
        x: Input sequence of complex numbers.

    Returns:
        List of complex DFT coefficients of the same length as ``x``.
    """
    N = len(x)
    if N == 0:
        return []
    result: List[complex] = []
    for k in range(N):
        s = 0j
        for n, xn in enumerate(x):
            angle = -2j * math.pi * k * n / N
            s += xn * cmath.exp(angle)
        result.append(s)
    return result


def idft(X: Sequence[complex]) -> List[complex]:
    """
    Compute the inverse discrete Fourier transform (IDFT).

    Args:
        X: Sequence of DFT coefficients.

    Returns:
        List of complex numbers representing the original time‑domain signal.
    """
    N = len(X)
    if N == 0:
        return []
    result: List[complex] = []
    for n in range(N):
        s = 0j
        for k, Xk in enumerate(X):
            angle = 2j * math.pi * k * n / N
            s += Xk * cmath.exp(angle)
        result.append(s / N)
    return result


def generate_square_wave(N: int, duty: float = 0.5) -> List[float]:
    """
    Generate a discrete square wave of length ``N`` with values +1 and -1.

    Args:
        N: Number of samples (must be non‑negative).
        duty: Fraction of the period where the signal is +1 (0 < duty ≤ 1).

    Returns:
        List of floats containing the square wave.
    """
    if N < 0:
        raise ValueError("N must be non‑negative")
    if not (0 < duty <= 1):
        raise ValueError("duty must be in (0, 1]")
    wave: List[float] = []
    for n in range(N):
        phase = n / N
        wave.append(1.0 if phase < duty else -1.0)
    return wave


def truncate_spectrum(X: Sequence[complex], num_terms: int) -> List[complex]:
    """
    Zero‑out all frequency components except the ``num_terms`` lowest
    positive frequencies and their symmetric negative counterparts.

    This mimics keeping only the first ``num_terms`` terms of a Fourier series.

    Args:
        X: Full DFT coefficient list.
        num_terms: Number of positive‑frequency terms to retain (excluding the DC term).

    Returns:
        A new list of coefficients with high‑frequency terms set to zero.
    """
    N = len(X)
    if num_terms < 0:
        raise ValueError("num_terms must be non‑negative")
    if N == 0:
        return []

    # Ensure we never keep more than the available frequencies.
    max_terms = N // 2
    num_terms = min(num_terms, max_terms)

    truncated = [0j] * N
    # DC component always kept
    truncated[0] = X[0]

    # Positive frequencies 1 .. num_terms
    for k in range(1, num_terms + 1):
        truncated[k] = X[k]
        # Symmetric negative frequency index
        neg_idx = N - k
        truncated[neg_idx] = X[neg_idx]

    # If N is even, the Nyquist frequency (k = N/2) is its own symmetric partner.
    if N % 2 == 0 and num_terms == N // 2:
        truncated[N // 2] = X[N // 2]

    return truncated


def fourier_series_approx(signal: Sequence[float], num_terms: int) -> List[float]:
    """
    Approximate ``signal`` using the first ``num_terms`` Fourier series terms.

    The algorithm:
        1. Compute the DFT of the input signal.
        2. Zero‑out all but the lowest ``num_terms`` frequencies.
        3. Perform the inverse DFT.
        4. Return the real part of the reconstructed signal.

    Args:
        signal: Real‑valued input sequence.
        num_terms: Number of positive‑frequency terms to retain.

    Returns:
        List of floats representing the approximated signal.
    """
    complex_signal = [complex(s, 0) for s in signal]
    X = dft(complex_signal)
    X_trunc = truncate_spectrum(X, num_terms)
    recon = idft(X_trunc)
    return [c.real for c in recon]
