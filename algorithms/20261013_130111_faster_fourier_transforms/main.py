from __future__ import annotations

import math
import random
import cmath
from typing import List

from .types import Signal, FFTLengthError
from .engine import fft, ifft, convolution

def _naive_dft(x: Signal) -> Signal:
    """Reference naive DFT implementation for testing correctness."""
    n = len(x)
    result: Signal = []
    for k in range(n):
        s = 0j
        for t in range(n):
            angle = -2j * cmath.pi * t * k / n
            s += x[t] * cmath.exp(angle)
        result.append(s)
    return result

def _assert_complex_almost_equal(a: complex, b: complex, eps: float = 1e-9) -> None:
    assert abs(a.real - b.real) <= eps and abs(a.imag - b.imag) <= eps, f"{a} != {b}"

def _assert_signal_almost_equal(a: Signal, b: Signal, eps: float = 1e-9) -> None:
    assert len(a) == len(b), f"Length mismatch {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        try:
            _assert_complex_almost_equal(x, y, eps)
        except AssertionError as e:
            raise AssertionError(f"Index {i}: {e}")

def test_fft_basic_cases() -> None:
    # Empty input
    assert fft([]) == []

    # Length‑1 input
    assert fft([3 + 4j]) == [3 + 4j]

    # Length‑2 input
    inp = [1 + 0j, -1 + 0j]
    expected = [0 + 0j, 2 + 0j]
    _assert_signal_almost_equal(fft(inp), expected)

    # Known 4‑point example: delta at index 1
    inp = [0j, 1j, 0j, 0j]
    # DFT of [0, i, 0, 0] = [i, -1, -i, 1]
    expected = [1j, -1 + 0j, -1j, 1 + 0j]
    _assert_signal_almost_equal(fft(inp), expected)

def test_fft_vs_naive() -> None:
    for n in (2, 4, 8, 16):
        for _ in range(10):
            signal = [complex(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(n)]
            fast = fft(signal)
            slow = _naive_dft(signal)
            _assert_signal_almost_equal(fast, slow)

def test_ifft_inverse_property() -> None:
    for n in (1, 2, 4, 8, 16):
        for _ in range(5):
            original = [complex(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(n)]
            transformed = fft(original)
            recovered = ifft(transformed)
            _assert_signal_almost_equal(original, recovered)

def test_convolution() -> None:
    # Simple convolution of two length‑4 sequences
    a = [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j]
    b = [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j]  # shift by one
    expected = [0 + 0j, 1 + 0j, 2 + 0j, 3 + 0j]  # circular shift
    result = convolution(a, b)
    _assert_signal_almost_equal(result, expected)

    # Random test against naive circular convolution
    for n in (2, 4, 8):
        for _ in range(5):
            a = [complex(random.uniform(-3, 3), random.uniform(-3, 3)) for _ in range(n)]
            b = [complex(random.uniform(-3, 3), random.uniform(-3, 3)) for _ in range(n)]
            # Naive circular convolution
            naive = [sum(a[(k - j) % n] * b[j] for j in range(n)) for k in range(n)]
            fast = convolution(a, b)
            _assert_signal_almost_equal(fast, naive)

def test_error_handling() -> None:
    try:
        fft([1, 2, 3])  # length 3 not power of two
    except FFTLengthError:
        pass
    else:
        raise AssertionError("FFTLengthError not raised for length 3")

    try:
        ifft([1, 2, 3, 4, 5])  # length 5 not power of two
    except FFTLengthError:
        pass
    else:
        raise AssertionError("FFTLengthError not raised for length 5")

    try:
        convolution([1, 2, 3, 4], [1, 2, 3])  # mismatched lengths
    except FFTLengthError:
        pass
    else:
        raise AssertionError("FFTLengthError not raised for mismatched lengths")

def run_all_tests() -> None:
    test_fft_basic_cases()
    test_fft_vs_naive()
    test_ifft_inverse_property()
    test_convolution()
    test_error_handling()
    print("All FFT unit tests passed successfully.")

if __name__ == '__main__':
    run_all_tests()
