import random
import unittest
import timeit
from typing import List

from core import transpose, Matrix


def naive_transpose(matrix: List[List[int]]) -> Matrix:
    """Simple O(m·n) transpose for correctness checking."""
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]


class TestCacheObliviousTranspose(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(0)

    def _random_matrix(self, rows: int, cols: int) -> List[List[int]]:
        return [[random.randint(-100, 100) for _ in range(cols)] for _ in range(rows)]

    def _assert_transpose_equal(self, mat: List[List[int]]) -> None:
        expected = naive_transpose(mat)
        result = transpose(mat)
        self.assertEqual(result, expected)

    def test_empty(self) -> None:
        self.assertEqual(transpose([]), [])

    def test_square(self) -> None:
        for n in [1, 2, 3, 8, 16, 31]:
            with self.subTest(size=n):
                mat = self._random_matrix(n, n)
                self._assert_transpose_equal(mat)

    def test_rectangular_more_rows(self) -> None:
        for rows, cols in [(5, 2), (10, 3), (17, 4)]:
            with self.subTest(rows=rows, cols=cols):
                mat = self._random_matrix(rows, cols)
                self._assert_transpose_equal(mat)

    def test_rectangular_more_cols(self) -> None:
        for rows, cols in [(2, 5), (3, 10), (4, 17)]:
            with self.subTest(rows=rows, cols=cols):
                mat = self._random_matrix(rows, cols)
                self._assert_transpose_equal(mat)

    def test_non_uniform_rows_error(self) -> None:
        mat = [[1, 2, 3], [4, 5]]
        with self.assertRaises(ValueError):
            transpose(mat)

    def test_threshold_effect(self) -> None:
        # Ensure algorithm works with a very small threshold (forces deep recursion)
        mat = self._random_matrix(7, 9)
        result = transpose(mat, threshold=1)
        self.assertEqual(result, naive_transpose(mat))

    def test_performance_small(self) -> None:
        # Very light benchmark to ensure function runs without error
        mat = self._random_matrix(64, 64)
        t = timeit.timeit(lambda: transpose(mat), number=10)
        self.assertLess(t, 1.0)  # Should be well under a second on typical hardware


def demo() -> None:
    """Simple demonstration printed to stdout."""
    rows, cols = 4, 6
    matrix = [[i * cols + j for j in range(cols)] for i in range(rows)]
    print("Original matrix:")
    for row in matrix:
        print(row)
    transposed = transpose(matrix)
    print("\nTransposed matrix:")
    for row in transposed:
        print(row)


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    print("\n--- Demo ---")
    demo()
