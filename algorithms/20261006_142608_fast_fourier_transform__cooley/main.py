from __future__ import annotations

import math
import random
from typing import List

from engine import fft, ifft
from custom_types import FFTResult, Signal

def _almost_equal(a: complex, b: complex, eps: float = 1e-9) -> bool:
    return abs(a.real - b.real) < eps and abs(a.imag - b.imag) < eps

def _assert_signal_close(sig1: Signal, sig2: Signal, eps: float = 1e-9) -> None:
    assert len(sig1) == len(sig2), f"Length mismatch: {len(sig1)} vs {len(sig2)}"
    for i, (x, y) in enumerate(zip(sig1, sig2)):
        assert _almost_equal(x, y, eps), f"Mismatch at index {i}: {x} vs {y}"

def test_fft_known_values() -> None:
    # Simple delta signal
    x: Signal = [1 + 0j, 0j, 0j, 0j]
    result: FFTResult = fft(x)
    expected: Signal = [1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j]
    _assert_signal_close(result.transformed, expected)

    # Two‑point alternating signal
    x = [1 + 0j, -1 + 0j, 1 + 0j, -1 + 0j]
    result = fft(x)
    expected = [0 + 0j, 0 + 0j, 4 + 0j, 0 + 0j]
    _assert_signal_close(result.transformed, expected)

def test_fft_ifft_roundtrip() -> None:
    # Random real signal of length 7 (will be padded to 8)
    random.seed(0)
    x = [random.random() for _ in range(7)]
    x_complex: Signal = [complex(v, 0) for v in x]

    forward = fft(x_complex)
    backward = ifft(forward.transformed)

    # Trim to original length and compare
    recovered = backward.transformed[: len(x_complex)]
    _assert_signal_close(recovered, x_complex, eps=1e-7)

def test_edge_cases() -> None:
    # Empty signal
    empty: Signal = []
    forward = fft(empty)
    assert forward.transformed == [], "FFT of empty list should be empty"
    backward = ifft(forward.transformed)
    assert backward.transformed == [], "IFFT of empty list should be empty"

    # Single element
    single = [3 + 4j]
    forward = fft(single)
    assert _almost_equal(forward.transformed[0], single[0]), "FFT of single element failed"
    backward = ifft(forward.transformed)
    assert _almost_equal(backward.transformed[0], single[0]), "IFFT of single element failed"

def run_all_tests() -> None:
    test_fft_known_values()
    test_fft_ifft_roundtrip()
    test_edge_cases()
    print("All FFT unit tests passed.")

if __name__ == "__main__":
    run_all_tests()
    # Demo: compute FFT of a sine wave
    N = 16
    freq = 3
    sine_wave: Signal = [complex(math.sin(2 * math.pi * freq * t / N), 0) for t in range(N)]
    fft_result = fft(sine_wave)
    print("\nFFT of sine wave (magnitude):")
    magnitudes = [abs(c) for c in fft_result.transformed]
    for i, mag in enumerate(magnitudes):
        print(f"Bin {i}: {mag:.3f}")
