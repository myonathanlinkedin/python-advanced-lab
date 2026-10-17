from __future__ import annotations

import random
import math
import sys
from typing import List

from .engine import fft, ifft, dft_naive, idft_naive
from .types import ComplexSignal

def _almost_equal(a: ComplexSignal, b: ComplexSignal, eps: float = 1e-9) -> bool:
    """Return True if two complex sequences are element‑wise equal within `eps`."""
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= eps for x, y in zip(a, b))

def _test_fft_vs_dft() -> None:
    """Compare FFT against the naïve DFT for several random power‑of‑two lengths."""
    for exp in range(0, 6):  # lengths 1,2,4,...,32
        n = 1 << exp
        signal = [complex(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)]
        fft_res = fft(signal)
        dft_res = dft_naive(signal)
        assert _almost_equal(fft_res, dft_res), f"FFT mismatch for n={n}"

def _test_ifft_recovery() -> None:
    """Verify that applying FFT then inverse FFT recovers the original signal."""
    for exp in range(0, 7):  # up to length 128
        n = 1 << exp
        signal = [complex(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(n)]
        transformed = fft(signal)
        recovered = ifft(transformed)
        assert _almost_equal(signal, recovered, eps=1e-7), f"Inverse FFT failed for n={n}"

def _test_edge_cases() -> None:
    """Test empty list, length‑1 list, and non‑power‑of‑two handling."""
    # Empty
    assert fft([]) == [], "FFT of empty list should be empty"
    assert ifft([]) == [], "iFFT of empty list should be empty"

    # Length 1
    single = [complex(3.14, -2.71)]
    assert _almost_equal(fft(single), single), "FFT length‑1 failed"
    assert _almost_equal(ifft(single), single), "iFFT length‑1 failed"

    # Non‑power‑of‑two should raise
    try:
        fft([1, 2, 3])  # length 3
        raise AssertionError("FFT did not raise on non‑power‑of‑two length")
    except ValueError:
        pass

    try:
        ifft([1, 2, 3, 4, 5])  # length 5
        raise AssertionError("iFFT did not raise on non‑power‑of‑two length")
    except ValueError:
        pass

def _demo() -> None:
    """Simple demonstration of forward and inverse FFT on a small signal."""
    signal: ComplexSignal = [complex(math.sin(2 * math.pi * i / 8), 0) for i in range(8)]
    print("Original signal:")
    for v in signal:
        print(f"{v:.4f}")

    transformed = fft(signal)
    print("\nFFT result:")
    for v in transformed:
        print(f"{v:.4f}")

    recovered = ifft(transformed)
    print("\nRecovered signal (after iFFT):")
    for v in recovered:
        print(f"{v:.4f}")

if __name__ == "__main__":
    # Run unit‑style assertions
    _test_edge_cases()
    _test_fft_vs_dft()
    _test_ifft_recovery()
    print("All unit tests passed.\n")

    # Run a short demo
    _demo()
