import math
import unittest
from typing import List

from core import (
    dft,
    idft,
    generate_square_wave,
    fourier_series_approx,
    truncate_spectrum,
)


def _max_abs_error(a: List[float], b: List[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


class TestFourierCore(unittest.TestCase):
    def test_dft_idft_identity(self):
        """Round‑trip DFT → IDFT should recover the original sequence."""
        for N in [1, 2, 3, 8, 15]:
            signal = [math.sin(2 * math.pi * i / N) for i in range(N)]
            X = dft([complex(s, 0) for s in signal])
            recon = idft(X)
            for original, recovered in zip(signal, recon):
                self.assertTrue(math.isclose(original, recovered.real, rel_tol=1e-9, abs_tol=1e-9))

    def test_known_constant_signal(self):
        """A constant signal should have only the DC component."""
        N = 8
        const = [5.0] * N
        X = dft([complex(v, 0) for v in const])
        # All coefficients except k=0 should be (approximately) zero.
        for k, coeff in enumerate(X):
            if k == 0:
                self.assertTrue(math.isclose(coeff.real, 5.0 * N, rel_tol=1e-9))
                self.assertTrue(math.isclose(coeff.imag, 0.0, abs_tol=1e-9))
            else:
                self.assertTrue(math.isclose(coeff.real, 0.0, abs_tol=1e-9))
                self.assertTrue(math.isclose(coeff.imag, 0.0, abs_tol=1e-9))

    def test_truncate_spectrum(self):
        """Truncating should keep the requested low frequencies and zero others."""
        N = 16
        signal = [math.cos(2 * math.pi * i / N) + 0.5 * math.cos(4 * math.pi * i / N) for i in range(N)]
        X = dft([complex(s, 0) for s in signal])
        X_trunc = truncate_spectrum(X, num_terms=1)  # keep only the first cosine term
        # The second harmonic should be removed (coefficients near zero)
        # Index for k=2 and its symmetric partner N-2
        self.assertTrue(math.isclose(X_trunc[2].real, 0.0, abs_tol=1e-9))
        self.assertTrue(math.isclose(X_trunc[N - 2].real, 0.0, abs_tol=1e-9))

    def test_square_wave_approximation_convergence(self):
        """Increasing the number of Fourier terms should reduce approximation error."""
        N = 64
        square = generate_square_wave(N, duty=0.5)
        errors: List[float] = []
        for terms in [1, 2, 4, 8, 16, 32]:
            approx = fourier_series_approx(square, num_terms=terms)
            err = _max_abs_error(square, approx)
            errors.append(err)
        # Errors must be non‑increasing
        for earlier, later in zip(errors, errors[1:]):
            self.assertLessEqual(later, earlier + 1e-12)  # allow tiny numerical noise

    def test_empty_inputs(self):
        """Functions should gracefully handle empty sequences."""
        self.assertEqual(dft([]), [])
        self.assertEqual(idft([]), [])
        self.assertEqual(truncate_spectrum([], 0), [])
        self.assertEqual(fourier_series_approx([], 5), [])


if __name__ == '__main__':
    unittest.main()
