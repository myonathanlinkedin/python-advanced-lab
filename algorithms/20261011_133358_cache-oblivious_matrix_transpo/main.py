import unittest
import random
import timeit
from typing import List

from core import cache_oblivious_transpose


def _transpose_reference(matrix: List[List[int]]) -> List[List[int]]:
    """Simple reference implementation using zip."""
    if not matrix:
        return []
    return [list(col) for col in zip(*matrix)]


class TestCacheObliviousTranspose(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(cache_oblivious_transpose([]), [])

    def test_single_element(self):
        mat = [[42]]
        self.assertEqual(cache_oblivious_transpose(mat), [[42]])

    def test_square(self):
        size = 8
        mat = [[i * size + j for j in range(size)] for i in range(size)]
        expected = _transpose_reference(mat)
        result = cache_oblivious_transpose(mat)
        self.assertEqual(result, expected)

    def test_rectangular_more_rows(self):
        rows, cols = 10, 4
        mat = [[i * cols + j for j in range(cols)] for i in range(rows)]
        expected = _transpose_reference(mat)
        result = cache_oblivious_transpose(mat)
        self.assertEqual(result, expected)

    def test_rectangular_more_cols(self):
        rows, cols = 3, 7
        mat = [[i * cols + j for j in range(cols)] for i in range(rows)]
        expected = _transpose_reference(mat)
        result = cache_oblivious_transpose(mat)
        self.assertEqual(result, expected)

    def test_non_uniform_rows_error(self):
        mat = [[1, 2, 3], [4, 5]]
        with self.assertRaises(ValueError):
            cache_oblivious_transpose(mat)

    def test_original_unchanged(self):
        mat = [[i + j for j in range(5)] for i in range(5)]
        copy = [row[:] for row in mat]
        _ = cache_oblivious_transpose(mat)
        self.assertEqual(mat, copy)

    def test_large_random(self):
        rows, cols = 128, 256
        random.seed(0)
        mat = [[random.randint(-1000, 1000) for _ in range(cols)] for _ in range(rows)]
        expected = _transpose_reference(mat)
        result = cache_oblivious_transpose(mat)
        self.assertEqual(result, expected)


def benchmark():
    rows, cols = 512, 512
    mat = [[i * cols + j for j in range(cols)] for i in range(rows)]
    t = timeit.timeit(lambda: cache_oblivious_transpose(mat), number=3)
    print(f"Cache‑oblivious transpose of {rows}×{cols} matrix: {t:.4f}s (average over 3 runs)")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Simple benchmark demonstration
    print("\nBenchmarking cache‑oblivious transpose:")
    benchmark()
