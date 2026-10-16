"""
Unit tests and simple benchmarks for the cache‑oblivious matrix transposition
implementation in ``core.py``.
"""

import random
import time
import unittest
from typing import List

from core import transpose, transpose_inplace

def _generate_matrix(rows: int, cols: int) -> List[List[int]]:
    """Create a rows×cols matrix filled with random integers."""
    rand = random.Random(0)  # deterministic seed for reproducibility
    return [[rand.randint(-1000, 1000) for _ in range(cols)] for _ in range(rows)]

def _naive_transpose(matrix: List[List[int]]) -> List[List[int]]:
    """Reference implementation using straightforward loops."""
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]

class TestCacheObliviousTranspose(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(transpose([]), [])

    def test_square(self):
        for n in range(1, 16):
            mat = _generate_matrix(n, n)
            expected = _naive_transpose(mat)
            result = transpose(mat)
            self.assertEqual(result, expected)

    def test_rectangular(self):
        sizes = [(2, 5), (5, 2), (3, 7), (7, 3), (8, 15), (15, 8)]
        for rows, cols in sizes:
            mat = _generate_matrix(rows, cols)
            expected = _naive_transpose(mat)
            result = transpose(mat)
            self.assertEqual(result, expected)

    def test_inplace_square(self):
        for n in range(1, 16):
            mat = _generate_matrix(n, n)
            expected = _naive_transpose(mat)
            transpose_inplace(mat)
            self.assertEqual(mat, expected)

    def test_inplace_error(self):
        mat = _generate_matrix(3, 4)
        with self.assertRaises(ValueError):
            transpose_inplace(mat)

def benchmark(rows: int, cols: int, repeats: int = 3) -> None:
    """Run a simple time benchmark comparing naive vs cache‑oblivious."""
    mat = _generate_matrix(rows, cols)
    # Warm‑up
    transpose(mat)
    start = time.perf_counter()
    for _ in range(repeats):
        transpose(mat)
    co_time = (time.perf_counter() - start) / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        _naive_transpose(mat)
    naive_time = (time.perf_counter() - start) / repeats

    print(
        f"{rows}×{cols}: cache‑oblivious={co_time:.6f}s, naive={naive_time:.6f}s, "
        f"speed‑up={naive_time / co_time:.2f}×"
    )

if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run a few benchmarks for illustration
    print("\nBenchmark results (average over 3 runs):")
    for size in [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]:
        benchmark(*size)
