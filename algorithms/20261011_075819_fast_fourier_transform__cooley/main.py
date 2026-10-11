"""
main.py
Unit‑test suite for the FFT implementation in core.py.
Running this module executes the tests and demonstrates a simple usage example.
"""

import math
import random
import unittest
from typing import List

from core import fft, ifft


def _naive_dft(x: List[complex]) -> List[complex]:
    """Reference DFT implementation with O(n²) complexity."""
    n = len(x)
    result: List[complex] = []
    for k in range(n):
        s = 0 + 0j
        for n_idx, val in enumerate(x):
            angle = -2 * math.pi * k * n_idx / n
            s += val * complex(math.cos(angle), math.sin(angle))
        result.append(s)
    return result


class TestFFT(unittest.TestCase):
    def assertComplexListAlmostEqual(self, a: List[complex], b: List[complex], places: int = 9):
        self.assertEqual(len(a), len(b), "Sequences differ in length")
        for i, (x, y) in enumerate(zip(a, b)):
            self.assertAlmostEqual(x.real, y.real, places=places,
                                   msg=f"Real part differs at index {i}")
            self.assertAlmostEqual(x.imag, y.imag, places=places,
                                   msg=f"Imag part differs at index {i}")

    def test_fft_length_one(self):
        x = [complex(3, -2)]
        self.assertComplexListAlmostEqual(fft(x), x)

    def test_fft_length_two(self):
        x = [1 + 0j, -1 + 0j]
        expected = [0 + 0j, 2 + 0j]  # DFT of [1, -1]
        self.assertComplexListAlmostEqual(fft(x), expected)

    def test_fft_known_sequence(self):
        # delta at index 1 for length 4 -> [1, -i, -1, i]
        x = [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j]
        expected = [1 + 0j, -1j, -1 + 0j, 1j]
        self.assertComplexListAlmostEqual(fft(x), expected)

    def test_fft_random_power_of_two(self):
        for n in (2, 4, 8, 16, 32):
            x = [complex(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)]
            fft_result = fft(x)
            dft_result = _naive_dft(x)
            self.assertComplexListAlmostEqual(fft_result, dft_result, places=7)

    def test_ifft_roundtrip(self):
        for n in (2, 4, 8, 16):
            original = [complex(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(n)]
            transformed = fft(original)
            recovered = ifft(transformed)
            self.assertComplexListAlmostEqual(recovered, original, places=7)

    def test_invalid_length_raises(self):
        with self.assertRaises(ValueError):
            fft([1, 2, 3])          # length 3 not power of two
        with self.assertRaises(ValueError):
            ifft([1, 2, 3, 4, 5])   # length 5 not power of two
        with self.assertRaises(ValueError):
            fft([])                 # empty input


if __name__ == '__main__':
    # Simple demonstration when the module is executed directly.
    demo_signal = [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j]  # delta at index 1, length 4
    print("Input signal:", demo_signal)
    transformed = fft(demo_signal)
    print("FFT:", transformed)
    recovered = ifft(transformed)
    print("Recovered signal (IFFT):", recovered)

    # Run the unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
