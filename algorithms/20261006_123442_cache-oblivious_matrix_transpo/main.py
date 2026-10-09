import unittest
import random
import time
from typing import List

from core import transpose


def _naive_transpose(matrix: List[List[int]]) -> List[List[int]]:
    """Simple O(m·n) transpose used as reference."""
    if not matrix:
        return []
    return [list(col) for col in zip(*matrix)]


class TestCacheObliviousTranspose(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(transpose([]), [])

    def test_single_element(self):
        self.assertEqual(transpose([[42]]), [[42]])

    def test_square_matrices(self):
        for size in range(1, 10):
            mat = [[i * size + j for j in range(size)] for i in range(size)]
            expected = _naive_transpose(mat)
            self.assertEqual(transpose(mat), expected)

    def test_rectangular_matrices(self):
        for rows in range(1, 8):
            for cols in range(1, 8):
                mat = [[random.randint(-100, 100) for _ in range(cols)] for _ in range(rows)]
                expected = _naive_transpose(mat)
                self.assertEqual(transpose(mat), expected)

    def test_unchanged_input(self):
        mat = [[i + j for j in range(5)] for i in range(3)]
        copy_mat = [row[:] for row in mat]
        _ = transpose(mat)
        self.assertEqual(mat, copy_mat)


def benchmark(size: int = 512, trials: int = 3) -> None:
    """Simple benchmark comparing naive vs cache‑oblivious transpose."""
    mat = [[random.random() for _ in range(size)] for _ in range(size)]

    # Warm‑up
    _ = transpose(mat)

    naive_times = []
    co_times = []

    for _ in range(trials):
        start = time.perf_counter()
        _ = _naive_transpose(mat)
        naive_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        _ = transpose(mat)
        co_times.append(time.perf_counter() - start)

    print(f"Matrix size: {size}×{size}")
    print(f"Naive avg time: {sum(naive_times) / trials:.6f}s")
    print(f"Cache‑oblivious avg time: {sum(co_times) / trials:.6f}s")


if __name__ == "__main__":
    # Run unit tests.
    unittest.main(exit=False)

    # Run a quick benchmark (optional, can be commented out).
    benchmark(size=256, trials=2)
