import unittest
import random
from typing import List, Tuple
from core import KDTree, euclidean_distance, Point

class TestKDTree(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(0)

    def test_single_point(self) -> None:
        pt = (1.0, 2.0)
        tree = KDTree([pt])
        self.assertEqual(tree.nearest((0.0, 0.0)), [(euclidean_distance((0.0, 0.0), pt), pt)])

    def test_two_dimensions(self) -> None:
        points = [(0, 0), (1, 1), (2, 2), (3, 3)]
        tree = KDTree(points)
        query = (1.2, 1.2)
        nearest = tree.nearest(query, k=2)
        expected = sorted(
            [(euclidean_distance(query, p), p) for p in points],
            key=lambda x: x[0]
        )[:2]
        self.assertEqual(nearest, expected)

    def test_high_dimension(self) -> None:
        dim = 5
        points = [tuple(random.random() for _ in range(dim)) for _ in range(50)]
        tree = KDTree(points)
        query = tuple(random.random() for _ in range(dim))
        # Brute‑force verification for k=3
        brute = sorted(
            [(euclidean_distance(query, p), p) for p in points],
            key=lambda x: x[0]
        )[:3]
        self.assertEqual(tree.nearest(query, k=3), brute)

    def test_k_equals_one(self) -> None:
        points = [(i, i * i) for i in range(10)]
        tree = KDTree(points)
        query = (5.1, 26.0)
        nearest = tree.nearest(query)
        brute = min(((euclidean_distance(query, p), p) for p in points), key=lambda x: x[0])
        self.assertEqual(nearest, [brute])

    def test_invalid_k(self) -> None:
        tree = KDTree([(0, 0)])
        with self.assertRaises(ValueError):
            tree.nearest((0, 0), k=0)

    def test_dimension_mismatch(self) -> None:
        tree = KDTree([(0, 0), (1, 1)])
        with self.assertRaises(ValueError):
            tree.nearest((0, 0, 0))

    def test_duplicate_points(self) -> None:
        points = [(1, 1), (1, 1), (2, 2)]
        tree = KDTree(points)
        query = (1, 1)
        nearest = tree.nearest(query, k=2)
        # Both duplicates should be returned with distance 0
        self.assertEqual(nearest, [(0.0, (1, 1)), (0.0, (1, 1))])

    def test_random_consistency(self) -> None:
        # Build tree once, query many times, compare with brute force each time
        points = [tuple(random.uniform(-100, 100) for _ in range(3)) for _ in range(200)]
        tree = KDTree(points)
        for _ in range(20):
            query = tuple(random.uniform(-100, 100) for _ in range(3))
            k = random.randint(1, 5)
            kd_res = tree.nearest(query, k=k)
            brute_res = sorted(
                [(euclidean_distance(query, p), p) for p in points],
                key=lambda x: x[0]
            )[:k]
            self.assertEqual(kd_res, brute_res)

if __name__ == "__main__":
    unittest.main()
