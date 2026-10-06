import random
import cmath
from typing import List
from .types import Signal
from .engine import fft, ifft, naive_dft

def _random_signal(length: int) -> Signal:
    return [complex(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(length)]

def _assert_close(a: Signal, b: Signal, eps: float = 1e-9) -> None:
    assert len(a) == len(b), "Signal lengths differ"
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) <= eps, f"Mismatch at index {i}: {x} vs {y}"

def run_tests() -> None:
    # Test 1: length 1 (trivial case)
    x1: Signal = [complex(0.5, -0.3)]
    _assert_close(fft(x1), naive_dft(x1))
    _assert_close(ifft(fft(x1)), x1)

    # Test 2: small power‑of‑two length
    x2 = [0j, 1+0j, 2+0j, 3+0j]
    _assert_close(fft(x2), naive_dft(x2))
    _assert_close(ifft(fft(x2)), x2)

    # Test 3: random length 8
    x8 = _random_signal(8)
    _assert_close(fft(x8), naive_dft(x8))
    _assert_close(ifft(fft(x8)), x8)

    # Test 4: random length 16
    x16 = _random_signal(16)
    _assert_close(fft(x16), naive_dft(x16))
    _assert_close(ifft(fft(x16)), x16)

    # Test 5: non‑power‑of‑two should raise
    try:
        fft([1, 2, 3])
        assert False, "Expected ValueError for non‑power‑of‑two length"
    except ValueError:
        pass

    try:
        ifft([1, 2, 3, 4, 5])
        assert False, "Expected ValueError for non‑power‑of‑two length"
    except ValueError:
        pass

    print("All FFT unit tests passed.")

def demo() -> None:
    # Simple demonstration with a sine wave
    import math
    N = 8
    freq = 1  # cycles per signal
    signal: Signal = [cmath.exp(2j * math.pi * freq * t / N) for t in range(N)]
    print("Original signal:")
    for v in signal:
        print(v)
    transformed = fft(signal)
    print("\nFFT result:")
    for v in transformed:
        print(v)
    recovered = ifft(transformed)
    print("\nRecovered signal (after IFFT):")
    for v in recovered:
        print(v)

if __name__ == "__main__":
    run_tests()
    demo()
