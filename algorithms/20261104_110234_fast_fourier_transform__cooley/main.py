import random
import unittest
import cmath
from typing import List, Sequence

from core import fft, ifft, convolve, is_power_of_two, next_power_of_two


def naive_dft(x: Sequence[complex]) -> List[complex]:
    """Direct O(N^2) DFT for verification."""
    n = len(x)
    result = []
    for k in range(n):
        s = 0j
        for t in range(n):
            angle = -2j * cmath.pi * k * t / n
            s += x[t] * cmath.exp(angle)
        result.append(s)
    return result


def naive_idft(X: Sequence[complex]) -> List[complex]:
    """Direct O(N^2) inverse DFT."""
    n = len(X)
    result = []
    for t in range(n):
        s = 0j
        for k in range(n):
            angle = 2j * cmath.pi * k * t / n
            s += X[k] * cmath.exp(angle)
        result.append(s / n)
    return result


def naive_convolve(a: Sequence[complex], b: Sequence[complex]) -> List[complex]:
    """Direct O(N*M) convolution."""
    m = len(a) + len(b) - 1
    result = [0j] * m
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            result[i + j] += av * bv
    return result


class TestFFT(unittest.TestCase):
    def test_is_power_of_two(self):
        self.assertTrue(is_power_of_two(1))
        self.assertTrue(is_power_of_two(2))
        self.assertTrue(is_power_of_two(1024))
        self.assertFalse(is_power_of_two(0))
        self.assertFalse(is_power_of_two(3))

    def test_next_power_of_two(self):
        self.assertEqual(next_power_of_two(1), 1)
        self.assertEqual(next_power_of_two(5), 8)
        self.assertEqual(next_power_of_two(16), 16)
        with self.assertRaises(ValueError):
            next_power_of_two(0)

    def test_fft_vs_naive(self):
        for n in [1, 2, 4, 8, 16]:
            for _ in range(5):
                data = [random.random() + random.random() * 1j for _ in range(n)]
                fft_res = fft(data)
                naive_res = naive_dft(data)
                for a, b in zip(fft_res, naive_res):
                    self.assertAlmostEqual(a.real, b.real, places=7)
                    self.assertAlmostEqual(a.imag, b.imag, places=7)

    def test_ifft_inverse(self):
        for n in [1, 2, 4, 8, 16, 32]:
            data = [random.random() + random.random() * 1j for _ in range(n)]
            transformed = fft(data)
            recovered = ifft(transformed)
            for a, b in zip(recovered, data):
                self.assertAlmostEqual(a.real, b.real, places=7)
                self.assertAlmostEqual(a.imag, b.imag, places=7)

    def test_ifft_vs_naive(self):
        for n in [1, 2, 4, 8]:
            data = [random.random() + random.random() * 1j for _ in range(n)]
            X = fft(data)
            inv = ifft(X)
            naive = naive_idft(X)
            for a, b in zip(inv, naive):
                self.assertAlmostEqual(a.real, b.real, places=7)
                self.assertAlmostEqual(a.imag, b.imag, places=7)

    def test_convolution(self):
        for _ in range(10):
            a_len = random.randint(1, 8)
            b_len = random.randint(1, 8)
            a = [random.random() + random.random() * 1j for _ in range(a_len)]
            b = [random.random() + random.random() * 1j for _ in range(b_len)]
            conv_fft = convolve(a, b)
            conv_naive = naive_convolve(a, b)
            self.assertEqual(len(conv_fft), len(conv_naive))
            for x, y in zip(conv_fft, conv_naive):
                self.assertAlmostEqual(x.real, y.real, places=7)
                self.assertAlmostEqual(x.imag, y.imag, places=7)

    def test_fft_invalid_length(self):
        with self.assertRaises(ValueError):
            fft([1, 2, 3])  # length 3 not power of two
        with self.assertRaises(ValueError):
            ifft([1, 2, 3, 4, 5])  # length 5 not power of two


if __name__ == "__main__":
    unittest.main()
