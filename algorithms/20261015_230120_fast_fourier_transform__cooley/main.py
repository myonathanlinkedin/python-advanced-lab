"""
main.py
Unit‑test suite and simple demonstration for the FFT implementation in core.py.
"""

import random
import time
import unittest
from typing import List

from core import fft, ifft, naive_dft, is_power_of_two, next_power_of_two


def _random_complex_list(n: int) -> List[complex]:
    """Generate a list of *n* random complex numbers with modest magnitude."""
    return [complex(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)]


class TestFFT(unittest.TestCase):
    def test_is_power_of_two(self) -> None:
        self.assertTrue(is_power_of_two(1))
        self.assertTrue(is_power_of_two(2))
        self.assertTrue(is_power_of_two(1024))
        self.assertFalse(is_power_of_two(0))
        self.assertFalse(is_power_of_two(3))
        self.assertFalse(is_power_of_two(1000))

    def test_next_power_of_two(self) -> None:
        self.assertEqual(next_power_of_two(1), 1)
        self.assertEqual(next_power_of_two(5), 8)
        self.assertEqual(next_power_of_two(1025), 2048)
        with self.assertRaises(ValueError):
            next_power_of_two(0)

    def test_fft_vs_naive(self) -> None:
        for n in (2, 4, 8, 16, 32):
            data = _random_complex_list(n)
            fast = fft(data)
            slow = naive_dft(data)
            for a, b in zip(fast, slow):
                self.assertAlmostEqual(a.real, b.real, places=7)
                self.assertAlmostEqual(a.imag, b.imag, places=7)

    def test_fft_invalid_length(self) -> None:
        with self.assertRaises(ValueError):
            fft([1, 2, 3])  # length 3 not power of two

    def test_ifft_roundtrip(self) -> None:
        for n in (2, 4, 8, 16, 64):
            original = _random_complex_list(n)
            transformed = fft(original)
            recovered = ifft(transformed)
            for a, b in zip(original, recovered):
                self.assertAlmostEqual(a.real, b.real, places=7)
                self.assertAlmostEqual(a.imag, b.imag, places=7)

    def test_ifft_invalid_length(self) -> None:
        with self.assertRaises(ValueError):
            ifft([1, 2, 3, 4, 5])  # length 5 not power of two

    def test_fft_performance(self) -> None:
        """A lightweight benchmark to ensure O(N log N) behaviour."""
        n = 2 ** 14  # 16384 points
        data = _random_complex_list(n)
        start = time.perf_counter()
        fft(data)
        duration = time.perf_counter() - start
        # Expect sub‑second on modest hardware; not a strict bound.
        self.assertLess(duration, 1.0, f"FFT took too long: {duration:.3f}s")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2)
    # Simple demo (executed only when not in test mode)
    demo_n = 8
    demo_signal = [complex(i, 0) for i in range(demo_n)]
    print("\nDemo signal:", demo_signal)
    transformed = fft(demo_signal)
    print("FFT:", transformed)
    recovered = ifft(transformed)
    print("Recovered signal (should match original):", recovered)
