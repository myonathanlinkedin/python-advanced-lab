import unittest
import random
import time
from typing import List
from core import transpose, Matrix

def naive_transpose(matrix: Matrix) -> Matrix:
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]

class TestCacheObliviousTranspose(unittest.TestCase):
    def _random_matrix(self, rows: int, cols: int) -> Matrix:
        return [[random.randint(-1000, 1000) for _ in range(cols)] for _ in range(rows)]

    def test_empty(self):
        self.assertEqual(transpose([]), [])
        self.assertEqual(transpose([[]]), [])

    def test_square_sizes(self):
        for n in [1, 2, 3, 4, 8, 16, 31, 64]:
            with self.subTest(size=n):
                m = self._random_matrix(n, n)
                self.assertEqual(transpose(m), naive_transpose(m))

    def test_rectangular(self):
        sizes = [(1, 5), (5, 1), (3, 7), (7, 3), (10, 2), (2, 10), (15, 20), (20, 15)]
        for rows, cols in sizes:
            with self.subTest(rows=rows, cols=cols):
                m = self._random_matrix(rows, cols)
                self.assertEqual(transpose(m), naive_transpose(m))

    def test_non_rectangular_error(self):
        m = [[1, 2], [3]]
        with self.assertRaises(ValueError):
            transpose(m)

    def test_original_unchanged(self):
        m = self._random_matrix(5, 4)
        copy = [row[:] for row in m]
        _ = transpose(m)
        self.assertEqual(m, copy)

    def test_large_random(self):
        rows, cols = 128, 96
        m = self._random_matrix(rows, cols)
        self.assertEqual(transpose(m), naive_transpose(m))

def benchmark():
    rows, cols = 1024, 1024
    matrix = [[i * cols + j for j in range(cols)] for i in range(rows)]
    start = time.perf_counter()
    _ = transpose(matrix)
    elapsed = time.perf_counter() - start
    print(f"Cache‑oblivious transpose of {rows}×{cols} took {elapsed:.4f}s")

if __name__ == '__main__':
    # Run unit tests
    unittest.main(exit=False)

    # Simple benchmark demonstration
    print("\nBenchmarking cache‑oblivious transpose:")
    benchmark()
