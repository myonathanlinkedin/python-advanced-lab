import unittest
import time
from typing import List

from core import AwkwardArray


def add(x: int, y: int) -> int:
    return x + y


class TestAwkwardArray(unittest.TestCase):
    def test_flatten_unflatten(self) -> None:
        nested = [[1, 2, 3], [4, 5], [], [6]]
        arr = AwkwardArray(nested)
        flat = arr.flatten()
        self.assertEqual(flat, [1, 2, 3, 4, 5, 6])
        reconstructed = arr.unflatten(flat)
        self.assertEqual(reconstructed.data, nested)

    def test_elementwise_add(self) -> None:
        a = AwkwardArray([[1, 2], [3, 4, 5]])
        b = AwkwardArray([[10, 20], [30, 40, 50]])
        c = a.gpu_compute(b, add)
        self.assertEqual(c.data, [[11, 22], [33, 44, 55]])

    def test_mismatched_shapes(self) -> None:
        a = AwkwardArray([[1, 2], [3]])
        b = AwkwardArray([[1, 2, 3], [4, 5]])
        with self.assertRaises(ValueError):
            a.gpu_compute(b, add)

    def test_large_array_performance(self) -> None:
        # Build a 1000 x 1000 ragged array (regular for simplicity)
        size = 500  # keep runtime modest for CI
        nested_a: List[List[int]] = [list(range(size)) for _ in range(size)]
        nested_b: List[List[int]] = [list(range(size, 0, -1)) for _ in range(size)]
        a = AwkwardArray(nested_a)
        b = AwkwardArray(nested_b)

        start = time.perf_counter()
        c = a.gpu_compute(b, add)
        gpu_time = time.perf_counter() - start

        # Sequential baseline
        start = time.perf_counter()
        flat_c = [x + y for row_a, row_b in zip(nested_a, nested_b) for x, y in zip(row_a, row_b)]
        seq = AwkwardArray(nested_a).unflatten(flat_c)
        seq_time = time.perf_counter() - start

        # Verify correctness
        self.assertEqual(c.data, seq.data)

        # Simple sanity check: parallel version should not be dramatically slower
        self.assertLess(gpu_time, seq_time * 5)


if __name__ == "__main__":
    unittest.main()
