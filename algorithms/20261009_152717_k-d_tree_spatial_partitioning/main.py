import unittest
import random
import math
from typing import List, Tuple
from core import KDTree, Point

def _brute_nearest(points: List[Point], target: Point) -> Tuple[Point, float]:
    """Reference implementation using linear scan."""
    best_pt = points[0]
    best_dist = math.dist(best_pt, target)
    for pt in points[1:]:
        d = math.dist(pt, target)
        if d < best_dist:
            best_dist = d
            best_pt = pt
    return best_pt, best_dist


class TestKDTree(unittest.TestCase):
    def test_single_point(self):
        pt = (1.0, 2.0, 3.0)
        tree = KDTree([pt])
        self.assertEqual(len(tree), 1)
        nn, d = tree.nearest((0.0, 0.0, 0.0))
        self.assertEqual(nn, pt)
        self.assertAlmostEqual(d, math.dist(pt, (0.0, 0.0, 0.0)))

    def test_two_points(self):
        pts = [(0.0, 0.0), (10.0, 0.0)]
        tree = KDTree(pts)
        nn, d = tree.nearest((6.0, 0.0))
        self.assertEqual(nn, (10.0, 0.0))
        self.assertAlmostEqual(d, 4.0)

    def test_random_2d(self):
        random.seed(0)
        pts = [(random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(200)]
        tree = KDTree(pts)
        for _ in range(50):
            query = (random.uniform(-100, 100), random.uniform(-100, 100))
            nn_tree, d_tree = tree.nearest(query)
            nn_brute, d_brute = _brute_nearest(pts, query)
            self.assertAlmostEqual(d_tree, d_brute, places=7)
            self.assertEqual(nn_tree, nn_brute)

    def test_random_3d(self):
        random.seed(1)
        pts = [
            (
                random.uniform(-50, 50),
                random.uniform(-50, 50),
                random.uniform(-50, 50),
            )
            for _ in range(150)
        ]
        tree = KDTree(pts)
        for _ in range(30):
            query = (
                random.uniform(-50, 50),
                random.uniform(-50, 50),
                random.uniform(-50, 50),
            )
            nn_tree, d_tree = tree.nearest(query)
            nn_brute, d_brute = _brute_nearest(pts, query)
            self.assertAlmostEqual(d_tree, d_brute, places=7)
            self.assertEqual(nn_tree, nn_brute)

    def test_dimension_mismatch(self):
        tree = KDTree([(0.0, 0.0), (1.0, 1.0)])
        with self.assertRaises(ValueError):
            tree.nearest((0.0, 0.0, 0.0))

    def test_empty_initialisation(self):
        with self.assertRaises(ValueError):
            KDTree([])

    def test_duplicate_points(self):
        pts = [(1.0, 1.0), (1.0, 1.0), (2.0, 2.0)]
        tree = KDTree(pts)
        nn, d = tree.nearest((1.0, 1.0))
        self.assertEqual(nn, (1.0, 1.0))
        self.assertAlmostEqual(d, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
