import unittest
from typing import List

from core import AwkwardArray, cuda_compute


class TestAwkwardArray(unittest.TestCase):
    def test_flatten_regular(self) -> None:
        arr = AwkwardArray([[1, 2], [3, 4]])
        self.assertEqual(arr.flatten(), [1, 2, 3, 4])

    def test_flatten_ragged(self) -> None:
        arr = AwkwardArray([[1, 2, 3], [4], [], [5, 6]])
        self.assertEqual(arr.flatten(), [1, 2, 3, 4, 5, 6])

    def test_shape_regular(self) -> None:
        arr = AwkwardArray([[1, 2], [3, 4]])
        self.assertEqual(arr.shape, (2, 2))

    def test_shape_ragged(self) -> None:
        arr = AwkwardArray([[1, 2, 3], [4], [], [5, 6]])
        # First dimension is 4, second is ragged -> -1
        self.assertEqual(arr.shape, (4, -1))

    def test_map_identity(self) -> None:
        arr = AwkwardArray([[1, 2], [3, 4]])
        result = arr.map(lambda x: x)
        self.assertEqual(result, arr)

    def test_map_increment(self) -> None:
        arr = AwkwardArray([[1, 2], [3, 4]])
        result = arr.map(lambda x: x + 1)  # type: ignore[arg-type]
        expected = AwkwardArray([[2, 3], [4, 5]])
        self.assertEqual(result, expected)

    def test_cuda_compute_sequential(self) -> None:
        arr = AwkwardArray([[1, 2], [3, 4]])
        result = cuda_compute(lambda x: x * x, arr, parallel=False)
        expected = AwkwardArray([[1, 4], [9, 16]])
        self.assertEqual(result, expected)

    def test_cuda_compute_parallel(self) -> None:
        arr = AwkwardArray([[1, 2, 3], [4, 5], [6]])
        result = cuda_compute(lambda x: x - 1, arr, parallel=True)
        expected = AwkwardArray([[0, 1, 2], [3, 4], [5]])
        self.assertEqual(result, expected)

    def test_cuda_compute_with_strings(self) -> None:
        arr = AwkwardArray([["a", "b"], ["c"]])
        result = cuda_compute(str.upper, arr, parallel=False)
        expected = AwkwardArray([["A", "B"], ["C"]])
        self.assertEqual(result, expected)


def benchmark_compute() -> None:
    import time

    size = 10_000
    nested: List[List[int]] = [list(range(i, i + 10)) for i in range(0, size, 10)]
    arr = AwkwardArray(nested)

    start = time.perf_counter()
    _ = cuda_compute(lambda x: x * 2, arr, parallel=False)
    seq_time = time.perf_counter() - start

    start = time.perf_counter()
    _ = cuda_compute(lambda x: x * 2, arr, parallel=True)
    par_time = time.perf_counter() - start

    print(f"Sequential time: {seq_time:.6f}s")
    print(f"Parallel time:   {par_time:.6f}s")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demonstration
    demo = AwkwardArray([[1, 2, 3], [4, 5], [6]])
    print("Original:", demo)
    squared = cuda_compute(lambda x: x ** 2, demo, parallel=True)
    print("Squared :", squared)
