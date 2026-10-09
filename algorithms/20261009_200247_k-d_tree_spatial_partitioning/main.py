import unittest
import random
import math
from typing import List, Tuple
from core import KDTree, Point

def _brute_knn(points: List[Point], query: Point, k: int) -> List[Tuple[Point, float]]:
    dists = [ (pt, math.sqrt(sum((a - b) ** 2 for a, b in zip(pt, query)))) for pt in points ]
    dists.sort(key=lambda pair: pair[1])
    return dists[:k]

class TestKDTree(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(0)

    def test_single_point(self) -> None:
        pt = (1.0, 2.0, 3.0)
        tree = KDTree([pt])
        self.assertEqual(tree.nearest(pt), [(pt, 0.0)])

    def test_two_points(self) -> None:
        pts = [(0.0, 0.0), (10.0, 0.0)]
        tree = KDTree(pts)
        self.assertEqual(tree.nearest((5.0, 0.0), k=2), [(pts[0], 5.0), (pts[1], 5.0)])

    def test_random_points(self) -> None:
        dim = 5
        n = 200
        points = [ tuple(random.uniform(-100, 100) for _ in range(dim)) for _ in range(n) ]
        tree = KDTree(points)
        for _ in range(20):
            query = tuple(random.uniform(-100, 100) for _ in range(dim))
            k = random.randint(1, 10)
            kd_res = tree.nearest(query, k=k)
            brute_res = _brute_knn(points, query, k)
            # compare distances with tolerance
            for (p1, d1), (p2, d2) in zip(kd_res, brute_res):
                self.assertEqual(p1, p2)
                self.assertAlmostEqual(d1, d2, places=7)

    def test_k_greater_than_points(self) -> None:
        pts = [(0.0, 0.0), (1.0, 1.0)]
        tree = KDTree(pts)
        res = tree.nearest((0.5, 0.5), k=5)
        self.assertEqual(len(res), 2)
        self.assertTrue(all(isinstance(d, float) for _, d in res))

    def test_invalid_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            KDTree([(1, 2), (1, 2, 3)])

        tree = KDTree([(0, 0)])
        with self.assertRaises(ValueError):
            tree.nearest((0, 0, 0))

    def test_invalid_k(self) -> None:
        tree = KDTree([(0, 0)])
        with self.assertRaises(ValueError):
            tree.nearest((0, 0), k=0)

    def test_empty_tree(self) -> None:
        with self.assertRaises(ValueError):
            KDTree([])

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
