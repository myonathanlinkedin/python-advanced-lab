import random
import math
import cmath
from typing import List

from custom_types import Signal, FFTResult
from engine import fft, ifft


def naive_dft(x: List[complex]) -> List[complex]:
    """Reference O(N²) DFT implementation for verification."""
    n = len(x)
    result = []
    for k in range(n):
        s = 0j
        for t in range(n):
            angle = -2j * math.pi * k * t / n
            s += x[t] * cmath.exp(angle)
        result.append(s)
    return result


def assert_complex_lists_close(a: List[complex], b: List[complex], eps: float = 1e-9) -> None:
    """Assert that two complex lists are element‑wise within `eps`."""
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x.real - y.real) <= eps and abs(x.imag - y.imag) <= eps, (
            f"Element {i} differs: {x} vs {y}"
        )


def test_fft_correctness():
    """Compare FFT against naive DFT for several power‑of‑two sizes."""
    for n in (1, 2, 4, 8, 16, 32):
        random.seed(0)  # deterministic
        samples = [complex(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)]
        sig = Signal(samples)
        fast = fft(sig).transformed
        slow = naive_dft(samples)
        assert_complex_lists_close(fast, slow)


def test_ifft_inverse():
    """Verify that IFFT(FFT(x)) recovers the original signal."""
    for n in (1, 2, 4, 8, 16, 64):
        random.seed(1)
        samples = [complex(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(n)]
        sig = Signal(samples)
        transformed = fft(sig)
        recovered = ifft(Signal(transformed.transformed)).transformed
        assert_complex_lists_close(samples, recovered, eps=1e-7)


def test_edge_cases():
    """Validate handling of empty and singleton inputs."""
    empty = Signal([])
    assert fft(empty).transformed == []
    assert ifft(empty).transformed == []

    singleton = Signal([3 + 4j])
    assert fft(singleton).transformed == [3 + 4j]
    assert ifft(singleton).transformed == [3 + 4j]


def test_invalid_length():
    """FFT must reject non‑power‑of‑two lengths."""
    bad = Signal([1, 2, 3])  # length 3
    try:
        fft(bad)
        assert False, "FFT should have raised ValueError for length 3"
    except ValueError:
        pass

    try:
        ifft(bad)
        assert False, "IFFT should have raised ValueError for length 3"
    except ValueError:
        pass


def demo():
    """Simple demonstration printing a forward and inverse transform."""
    samples = [cmath.exp(2j * math.pi * i / 8) for i in range(8)]
    sig = Signal(samples)
    transformed = fft(sig).transformed
    recovered = ifft(Signal(transformed)).transformed
    print("Original samples:")
    for s in samples:
        print(f"{s:.4f}")
    print("\nFFT:")
    for s in transformed:
        print(f"{s:.4f}")
    print("\nRecovered (IFFT):")
    for s in recovered:
        print(f"{s:.4f}")


if __name__ == '__main__':
    test_fft_correctness()
    test_ifft_inverse()
    test_edge_cases()
    test_invalid_length()
    demo()
    print("\nAll tests passed.")
