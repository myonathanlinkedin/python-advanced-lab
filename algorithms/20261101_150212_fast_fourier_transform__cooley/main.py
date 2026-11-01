from __future__ import annotations
import unittest
import math
from typing import List
from .engine import fft, ifft
from .types import Signal, FFTResult

class TestFFT(unittest.TestCase):
    def assertComplexListAlmostEqual(self, a: List[complex], b: List[complex], places: int = 7):
        self.assertEqual(len(a), len(b), "Lists differ in length")
        for i, (x, y) in enumerate(zip(a, b)):
            self.assertAlmostEqual(x.real, y.real, places=places,
                                   msg=f"Real part differs at index {i}")
            self.assertAlmostEqual(x.imag, y.imag, places=places,
                                   msg=f"Imag part differs at index {i}")

    def test_fft_known_values(self):
        # Simple length‑4 test: DFT of [1, 0, 0, 0] should be all ones
        input_signal: Signal = Signal([1+0j, 0+0j, 0+0j, 0+0j])
        expected: FFTResult = FFTResult([1+0j, 1+0j, 1+0j, 1+0j])
        result = fft(input_signal)
        self.assertComplexListAlmostEqual(result, expected)

        # DFT of [0, 1, 0, 0] yields [1, -i, -1, i]
        input_signal = Signal([0+0j, 1+0j, 0+0j, 0+0j])
        expected = FFTResult([
            1+0j,
            0-1j,
            -1+0j,
            0+1j
        ])
        result = fft(input_signal)
        self.assertComplexListAlmostEqual(result, expected)

    def test_ifft_inverse(self):
        # Random length‑8 signal
        original: Signal = Signal([complex(math.sin(i), math.cos(i)) for i in range(8)])
        transformed = fft(original)
        recovered = ifft(transformed)
        self.assertComplexListAlmostEqual(recovered, original, places=6)

    def test_non_power_of_two(self):
        with self.assertRaises(ValueError):
            fft([1, 2, 3])  # length 3 not power of two
        with self.assertRaises(ValueError):
            ifft([1, 2, 3, 4, 5])  # length 5 not power of two

    def test_fft_ifft_consistency(self):
        # Verify that applying fft then ifft returns the original (within tolerance)
        for n in (2, 4, 8, 16):
            signal = Signal([complex(i, -i) for i in range(n)])
            self.assertComplexListAlmostEqual(ifft(fft(signal)), signal, places=6)

if __name__ == '__main__':
    unittest.main()
