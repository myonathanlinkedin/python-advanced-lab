import unittest
import random
import math
import cmath
import time
from core import fft, ifft

class TestFFT(unittest.TestCase):
    def dft_manual(self, x: list[complex]) -> list[complex]:
        n = len(x)
        return [
            sum(x[k] * cmath.exp(-2j * math.pi * j * k / n) for k in range(n))
            for j in range(n)
        ]

    def test_small_array(self):
        x = [0, 1, 2, 3]
        expected = self.dft_manual(x)
        result = fft([complex(v) for v in x])
        for a, b in zip(expected, result):
            self.assertAlmostEqual(a.real, b.real, places=6)
            self.assertAlmostEqual(a.imag, b.imag, places=6)

    def test_random_arrays(self):
        for n in [8, 16, 32]:
            for _ in range(5):
                x = [random.uniform(-1, 1) + 1j * random.uniform(-1, 1) for _ in range(n)]
                expected = self.dft_manual(x)
                result = fft(x)
                for a, b in zip(expected, result):
                    self.assertAlmostEqual(a.real, b.real, places=5)
                    self.assertAlmostEqual(a.imag, b.imag, places=5)

    def test_ifft_roundtrip(self):
        for n in [8, 16, 32]:
            for _ in range(5):
                x = [random.uniform(-1, 1) + 1j * random.uniform(-1, 1) for _ in range(n)]
                y = ifft(fft(x))
                for a, b in zip(x, y):
                    self.assertAlmostEqual(a.real, b.real, places=5)
                    self.assertAlmostEqual(a.imag, b.imag, places=5)

class BenchmarkFFT(unittest.TestCase):
    def benchmark(self, n: int):
        x = [complex(random.random(), random.random()) for _ in range(n)]
        start = time.perf_counter()
        fft(x)
        return time.perf_counter() - start

    def test_benchmark(self):
        times = {}
        for n in [2**10, 2**12, 2**14]:
            t = self.benchmark(n)
            times[n] = t
        # Ensure times are increasing with size
        self.assertTrue(times[2**10] < times[2**12] < times[2**14])

if __name__ == "__main__":
    # Demo: FFT of a simple impulse
    impulse = [0] * 8
    impulse[0] = 1
    spectrum = fft([complex(v) for v in impulse])
    print("FFT of impulse:")
    for i, val in enumerate(spectrum):
        print(f"Index {i}: {val:.4f}")

    # Run tests
    unittest.main(exit=False)
