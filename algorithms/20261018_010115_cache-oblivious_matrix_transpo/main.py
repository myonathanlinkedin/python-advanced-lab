import unittest
import random
import time
from typing import List
from core import transpose, Matrix

def naive_transpose(matrix: Matrix) -> Matrix:
    if not matrix:
        return []
    return [list(col) for col in zip(*matrix)]

class TestCacheObliviousTranspose(unittest.TestCase):
    def _random_matrix(self, rows: int, cols: int) -> Matrix:
        return [[random.randint(-1000, 1000) for _ in range(cols)] for _ in range(rows)]

    def test_empty(self):
        self.assertEqual(transpose([]), [])
        self.assertEqual(transpose([[]]), [])

    def test_square(self):
        for n in [1, 2, 3, 4, 8, 16, 31]:
            with self.subTest(n=n):
                m = self._random_matrix(n, n)
                self.assertEqual(transpose(m), naive_transpose(m))

    def test_rectangular(self):
        sizes = [(1, 5), (5, 1), (3, 7), (7, 3), (10, 15), (15, 10)]
        for rows, cols in sizes:
            with self.subTest(rows=rows, cols=cols):
                m = self._random_matrix(rows, cols)
                self.assertEqual(transpose(m), naive_transpose(m))

    def test_double_transpose(self):
        for rows, cols in [(4, 6), (7, 2), (12, 12)]:
            m = self._random_matrix(rows, cols)
            t = transpose(m)
            tt = transpose(t)
            self.assertEqual(tt, m)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            transpose([[1, 2], [3]])  # ragged rows

    def test_threshold_effect(self):
        # Ensure algorithm works for a very small threshold (forces deep recursion)
        m = self._random_matrix(8, 9)
        self.assertEqual(transpose(m, threshold=1), naive_transpose(m))

class Benchmark:
    @staticmethod
    def run():
        sizes = [(256, 256), (512, 512), (1024, 1024)]
        print("Benchmarking cache‑oblivious transpose vs naive")
        for rows, cols in sizes:
            mat = [[random.random() for _ in range(cols)] for _ in range(rows)]
            # Cache‑oblivious
            start = time.perf_counter()
            transpose(mat)
            co_time = time.perf_counter() - start
            # Naive
            start = time.perf_counter()
            naive_transpose(mat)
            naive_time = time.perf_counter() - start
            print(f"{rows}×{cols}: CO={co_time:.6f}s, naive={naive_time:.6f}s")

if __name__ == "__main__":
    unittest.main(exit=False)
    Benchmark.run()
