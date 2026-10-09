import random
import math
import cmath
from typing import List

from .engine import fft, ifft, _naive_dft, FFTInputError

def _assert_almost_equal(a: List[complex], b: List[complex], eps: float = 1e-9) -> None:
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if abs(x.real - y.real) > eps or abs(x.imag - y.imag) > eps:
            raise AssertionError(f"Element {i} differs: {x} vs {y}")

def _test_fft_correctness() -> None:
    # Test several powers of two
    for power in range(1, 5):  # lengths 2,4,8,16
        N = 2 ** power
        signal = [random.uniform(-1, 1) + random.uniform(-1, 1) * 1j for _ in range(N)]
        fft_result = fft(signal)
        dft_result = _naive_dft(signal)
        _assert_almost_equal(fft_result, dft_result)

def _test_ifft_inverse() -> None:
    for power in range(1, 5):
        N = 2 ** power
        signal = [random.uniform(-5, 5) + random.uniform(-5, 5) * 1j for _ in range(N)]
        spectrum = fft(signal)
        recovered = ifft(spectrum)
        _assert_almost_equal(recovered, signal)

def _test_invalid_lengths() -> None:
    # Length zero
    try:
        fft([])
        raise AssertionError("FFT did not raise on empty input")
    except FFTInputError:
        pass
    # Non‑power‑of‑two length
    try:
        fft([1, 2, 3])
        raise AssertionError("FFT did not raise on non‑power‑of‑two input")
    except FFTInputError:
        pass
    # Same for IFFT
    try:
        ifft([1, 2, 3, 4, 5])
        raise AssertionError("IFFT did not raise on non‑power‑of‑two input")
    except FFTInputError:
        pass

def _demo() -> None:
    # Simple demonstration with a known signal
    signal = [cmath.exp(2j * cmath.pi * k / 8) for k in range(8)]
    print("Original signal:")
    for v in signal:
        print(v)
    spectrum = fft(signal)
    print("\nFFT spectrum:")
    for v in spectrum:
        print(v)
    recovered = ifft(spectrum)
    print("\nRecovered signal (should match original):")
    for v in recovered:
        print(v)

if __name__ == '__main__':
    _test_fft_correctness()
    _test_ifft_inverse()
    _test_invalid_lengths()
    print("All unit tests passed.\n")
    _demo()
