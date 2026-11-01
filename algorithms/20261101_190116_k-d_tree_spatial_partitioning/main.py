import unittest
import random
import math
from typing import List, Tuple
from core import KDTree, Point


def brute_force_nn(points: List[Point], target: Point) -> Tuple[Point, float]:
    best = points[0]
    best_dist = math.dist(best, target)
    for p in points[1:]:
        d = math.dist(p, target)
        if d < best_dist:
            best, best_dist = p, d
    return best, best_dist


class TestKDTree(unittest.TestCase):
    def test_basic_2d(self) -> None:
        pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]
        tree = KDTree(pts)
        q = (1.2, 1.1)
        nn, dist = tree.nearest(q)
        bf_nn, bf_dist = brute_force_nn(pts, q)
        self.assertEqual(nn, bf_nn)
        self.assertAlmostEqual(dist, bf_dist, places=7)

    def test_random_3d(self) -> None:
        random.seed(0)
        pts = [tuple(random.uniform(-100, 100) for _ in range(3)) for _ in range(200)]
        tree = KDTree(pts)
        for _ in range(20):
            q = tuple(random.uniform(-100, 100) for _ in range(3))
            nn, dist = tree.nearest(q)
            bf_nn, bf_dist = brute_force_nn(pts, q)
            self.assertEqual(nn, bf_nn)
            self.assertAlmostEqual(dist, bf_dist, places=7)

    def test_duplicate_points(self) -> None:
        pts = [(1.0, 2.0), (1.0, 2.0), (3.0, 4.0)]
        tree = KDTree(pts)
        q = (1.0, 2.0)
        nn, dist = tree.nearest(q)
        self.assertEqual(nn, (1.0, 2.0))
        self.assertEqual(dist, 0.0)

    def test_high_dimensional(self) -> None:
        dim = 5
        pts = [tuple(i + j * 0.1 for i in range(dim)) for j in range(30)]
        tree = KDTree(pts)
        q = tuple(15.3 + i * 0.05 for i in range(dim))
        nn, dist = tree.nearest(q)
        bf_nn, bf_dist = brute_force_nn(pts, q)
        self.assertEqual(nn, bf_nn)
        self.assertAlmostEqual(dist, bf_dist, places=7)

    def test_empty_tree_error(self) -> None:
        with self.assertRaises(ValueError):
            KDTree([])

    def test_dimension_mismatch(self) -> None:
        tree = KDTree([(0, 0), (1, 1)])
        with self.assertRaises(ValueError):
            tree.nearest((0, 0, 0))

    def test_single_point(self) -> None:
        pt = (42.0, -7.5)
        tree = KDTree([pt])
        nn, dist = tree.nearest((0.0, 0.0))
        self.assertEqual(nn, pt)
        self.assertAlmostEqual(dist, math.dist(pt, (0.0, 0.0)), places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
