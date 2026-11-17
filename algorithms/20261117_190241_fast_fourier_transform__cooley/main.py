import math
import random
from typing import List

from engine import fft, ifft, _naive_dft

def allclose(a: List[complex], b: List[complex], tol: float = 1e-9) -> bool:
    """Return True if all elements of a and b are close within tolerance."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if not (math.isclose(x.real, y.real, abs_tol=tol) and
                math.isclose(x.imag, y.imag, abs_tol=tol)):
            return False
    return True

def test_fft_basic():
    # Test known small cases
    assert allclose(fft([0, 1, 0, 0]), [1, -1j, -1, 1j])
    assert allclose(fft([1, 0, 1, 0]), [2, 0, 2, 0])
    assert allclose(fft([1, 2, 3, 4]), [10, -2+2j, -2, -2-2j])
    # Length 1
    assert allclose(fft([5]), [5])
    # Length 2
    assert allclose(fft([1, 2]), [3, -1])

def test_ifft_basic():
    # Inverse of known FFTs
    assert allclose(ifft([1, -1j, -1, 1j]), [0, 1, 0, 0])
    assert allclose(ifft([2, 0, 2, 0]), [1, 0, 1, 0])
    assert allclose(ifft([10, -2+2j, -2, -2-2j]), [1, 2, 3, 4])
    # Inverse of delta
    assert allclose(ifft([1, 0, 0, 0]), [0.25, 0.25, 0.25, 0.25])
    # Length 1
    assert allclose(ifft([5]), [5])
    # Length 2
    assert allclose(ifft([3, -1]), [1, 2])

def test_fft_ifft_consistency():
    for n in [1, 2, 4, 8, 16, 32, 64]:
        data = [random.uniform(-10, 10) + random.uniform(-10, 10)*1j for _ in range(n)]
        fft_res = fft(data)
        ifft_res = ifft(fft_res)
        assert allclose(ifft_res, data), f"Mismatch for n={n}"

def test_naive_vs_fft():
    for n in [1, 2, 4, 8, 16]:
        data = [random.uniform(-5, 5) + random.uniform(-5, 5)*1j for _ in range(n)]
        assert allclose(fft(data), _naive_dft(data)), f"Naive vs FFT mismatch for n={n}"

def test_power_of_two_error():
    try:
        fft([1, 2, 3])  # length 3 not power of two
    except ValueError:
        pass
    else:
        raise AssertionError("fft did not raise ValueError for non-power-of-two length")
    try:
        ifft([1, 2, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("ifft did not raise ValueError for non-power-of-two length")

def test_edge_cases():
    # Empty input should raise ValueError
    try:
        fft([])
    except ValueError:
        pass
    else:
        raise AssertionError("fft did not raise ValueError for empty input")
    try:
        ifft([])
    except ValueError:
        pass
    else:
        raise AssertionError("ifft did not raise ValueError for empty input")

def run_all_tests():
    test_fft_basic()
    test_ifft_basic()
    test_fft_ifft_consistency()
    test_naive_vs_fft()
    test_power_of_two_error()
    test_edge_cases()
    print("All tests passed.")

if __name__ == "__main__":
    run_all_tests()
    # Demo: FFT of a simple sequence
    seq = [1, 2, 3, 4]
    print(f"Input sequence: {seq}")
    fft_result = fft(seq)
    print(f"FFT result: {fft_result}")
    ifft_result = ifft(fft_result)
    print(f"Inverse FFT (should match input): {ifft_result}")
