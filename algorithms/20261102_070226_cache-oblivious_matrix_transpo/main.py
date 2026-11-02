import unittest
import random
import time
from typing import List
from core import transpose

class TestCacheObliviousTranspose(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(transpose([]), [])

    def test_single_element(self):
        self.assertEqual(transpose([[42]]), [[42]])

    def test_one_row(self):
        mat = [[1, 2, 3, 4]]
        expected = [[1], [2], [3], [4]]
        self.assertEqual(transpose(mat), expected)

    def test_one_column(self):
        mat = [[1], [2], [3], [4]]
        expected = [[1, 2, 3, 4]]
        self.assertEqual(transpose(mat), expected)

    def test_square(self):
        mat = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        expected = [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
        ]
        self.assertEqual(transpose(mat), expected)

    def test_rectangular_more_rows(self):
        mat = [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ]
        expected = [
            [1, 3, 5, 7],
            [2, 4, 6, 8],
        ]
        self.assertEqual(transpose(mat), expected)

    def test_rectangular_more_cols(self):
        mat = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
        expected = [
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 8],
        ]
        self.assertEqual(transpose(mat), expected)

    def test_random_rectangular(self):
        for rows in range(1, 20):
            for cols in range(1, 20):
                mat = [[random.randint(-1000, 1000) for _ in range(cols)] for _ in range(rows)]
                # Naïve transpose for verification
                expected = [[mat[r][c] for r in range(rows)] for c in range(cols)]
                self.assertEqual(transpose(mat), expected)

    def test_non_rectangular_error(self):
        mat = [[1, 2], [3]]
        with self.assertRaises(ValueError):
            transpose(mat)

    def test_performance_small(self):
        # Simple timing to ensure the function returns quickly for modest sizes
        mat = [[i * j for j in range(64)] for i in range(64)]
        start = time.perf_counter()
        _ = transpose(mat)
        duration = time.perf_counter() - start
        self.assertLess(duration, 0.5)  # Should be well under half a second on typical hardware

if __name__ == '__main__':
    unittest.main()
