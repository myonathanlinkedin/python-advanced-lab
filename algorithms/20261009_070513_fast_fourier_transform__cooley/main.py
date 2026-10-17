from __future__ import annotations

import random
import math
import cmath
from typing import List

from .engine import fft, ifft, naive_dft, FFTResult

def _almost_equal(a: List[complex], b: List[complex], eps: float = 1e-9) -> bool:
    """Return True if two complex vectors are equal within tolerance."""
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= eps for x, y in zip(a, b))

def _test_fft_correctness():
    """Compare FFT against naive DFT for several random inputs."""
    for n in (1, 2, 4, 8, 16, 32):
        for _ in range(5):
            signal = [random.uniform(-1, 1) + random.uniform(-1, 1) * 1j for _ in range(n)]
            fft_res = fft(signal).values
            dft_res = naive_dft(signal)
            assert _almost_equal(list(fft_res), dft_res), f"FFT mismatch for n={n}"

def _test_ifft_inverse():
    """Verify that ifft(fft(x)) recovers the original signal."""
    for n in (1, 2, 4, 8, 16, 32):
        signal = [random.uniform(-5, 5) + random.uniform(-5, 5) * 1j for _ in range(n)]
        transformed = fft(signal).values
        recovered = ifft(transformed).values
        assert _almost_equal(list(recovered), signal), f"IFFT inverse failed for n={n}"

def _test_invalid_lengths():
    """Ensure non‑power‑of‑two inputs raise ValueError."""
    for n in (0, 3, 5, 6, 7, 9, 10):
        signal = [0j] * n
        try:
            fft(signal)
        except ValueError:
            pass
        else:
            assert False, f"fft should have raised for length {n}"
        try:
            ifft(signal)
        except ValueError:
            pass
        else:
            assert False, f"ifft should have raised for length {n}"

def _demo():
    """Simple demonstration of FFT on a sinusoidal signal."""
    N = 8
    freq = 1  # cycles per N samples
    signal = [cmath.exp(2j * math.pi * freq * t / N) for t in range(N)]
    print("Input signal:")
    for v in signal:
        print(f"{v:.3f}")
    transformed = fft(signal).values
    print("\nFFT output (magnitude):")
    for v in transformed:
        print(f"{abs(v):.3f}")

if __name__ == '__main__':
    _test_fft_correctness()
    _test_ifft_inverse()
    _test_invalid_lengths()
    print("All unit tests passed.\n")
    _demo()
