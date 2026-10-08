import random
import time
import unittest
from typing import List

from core import RectangularMatrix, multiply_matrices, Matrix


def _random_matrix(rows: int, cols: int, low: int = -10, high: int = 10) -> Matrix:
    return [[random.randint(low, high) for _ in range(cols)] for _ in range(rows)]


class TestMatrixMultiplication(unittest.TestCase):
    def test_basic(self) -> None:
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[7, 8], [9, 10], [11, 12]]
        expected = [[58, 64], [139, 154]]
        self.assertEqual(multiply_matrices(a, b), expected)

        ma = RectangularMatrix(a)
        mb = RectangularMatrix(b)
        self.assertEqual((ma * mb).data, expected)

    def test_random_small(self) -> None:
        for _ in range(20):
            m, k, n = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
            a = _random_matrix(m, k)
            b = _random_matrix(k, n)
            # Compute expected using a straightforward Python comprehension
            expected = [[sum(a[i][p] * b[p][j] for p in range(k)) for j in range(n)] for i in range(m)]
            self.assertEqual(multiply_matrices(a, b), expected)

    def test_invalid_dimensions(self) -> None:
        a = [[1, 2], [3, 4]]
        b = [[5, 6, 7]]
        with self.assertRaises(ValueError):
            multiply_matrices(a, b)
        with self.assertRaises(ValueError):
            RectangularMatrix(a) * RectangularMatrix(b)

    def test_zero_sized(self) -> None:
        a: Matrix = [[1]]
        b: Matrix = [[2]]
        self.assertEqual(multiply_matrices(a, b), [[2]])
        # Empty matrices are not allowed
        with self.assertRaises(ValueError):
            multiply_matrices([], b)
        with self.assertRaises(ValueError):
            multiply_matrices(a, [])

    def test_immutability(self) -> None:
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        m = RectangularMatrix(a)
        _ = m * RectangularMatrix(b)
        # Original data should stay unchanged
        self.assertEqual(a, [[1, 2], [3, 4]])
        self.assertEqual(b, [[5, 6], [7, 8]])


def benchmark(rows: int, inner: int, cols: int, repeats: int = 3) -> None:
    """Simple benchmark printing average time for a single multiplication."""
    a = _random_matrix(rows, inner)
    b = _random_matrix(inner, cols)
    # Warm‑up
    multiply_matrices(a, b)
    times: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        multiply_matrices(a, b)
        times.append(time.perf_counter() - start)
    avg = sum(times) / repeats
    print(
        f"Benchmark {rows}x{inner} * {inner}x{cols}: "
        f"average {avg * 1_000:.2f} ms over {repeats} runs"
    )


def demo() -> None:
    """Run a small demonstration of rectangular matrix multiplication."""
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8], [9, 10], [11, 12]]
    print("Matrix A:", a)
    print("Matrix B:", b)
    c = multiply_matrices(a, b)
    print("A × B =", c)


if __name__ == "__main__":
    # Execute unit tests first; continue to demo and benchmark regardless of outcome.
    unittest.main(exit=False)
    demo()
    # Example benchmark (moderate size, safe for quick execution)
    benchmark(100, 200, 150, repeats=5)
