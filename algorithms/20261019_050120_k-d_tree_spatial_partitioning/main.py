from __future__ import annotations
import random
import math
from typing import List, Tuple

from .types import Point
from .engine import KDTree


def _brute_force_nearest(points: List[Point], target: Point) -> Tuple[Point, float]:
    """Reference implementation using linear scan."""
    best_point = points[0]
    best_dist = math.inf
    for p in points:
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(p, target)))
        if d < best_dist:
            best_dist = d
            best_point = p
    return best_point, best_dist


def _assert_nearest(tree: KDTree, points: List[Point], target: Point) -> None:
    """Validate that KDTree.nearest matches brute‑force result."""
    kd_point, kd_dist = tree.nearest(target)
    bf_point, bf_dist = _brute_force_nearest(points, target)
    assert kd_point == bf_point, f"Nearest point mismatch: {kd_point} vs {bf_point}"
    assert math.isclose(kd_dist, bf_dist, rel_tol=1e-9), f"Distance mismatch: {kd_dist} vs {bf_dist}"


def _run_basic_tests() -> None:
    # 1‑D trivial case
    points_1d = [(0.0,), (5.0,), (10.0,)]
    tree_1d = KDTree(points_1d)
    _assert_nearest(tree_1d, points_1d, (6.0,))

    # 2‑D simple geometry
    points_2d = [(0, 0), (1, 1), (2, 2), (5, 5)]
    tree_2d = KDTree(points_2d)
    _assert_nearest(tree_2d, points_2d, (3, 3))

    # Single point tree
    single = [(42.0, -7.0)]
    tree_single = KDTree(single)
    assert tree_single.nearest((0, 0))[0] == (42.0, -7.0)

    # Empty tree should raise
    empty = KDTree([])
    try:
        empty.nearest((0, 0))
        assert False, "Expected ValueError for empty tree"
    except ValueError:
        pass


def _run_randomized_tests() -> None:
    random.seed(0)
    for dim in (1, 2, 3, 5):
        for n in (10, 50, 200):
            points = [tuple(random.uniform(-100, 100) for _ in range(dim)) for _ in range(n)]
            tree = KDTree(points)
            for _ in range(20):
                target = tuple(random.uniform(-100, 100) for _ in range(dim))
                _assert_nearest(tree, points, target)


def _run_insertion_test() -> None:
    # Start with empty tree and insert incrementally, checking after each insertion
    points = [(1.0, 2.0), (3.0, 4.0), (-1.0, -2.0), (0.0, 0.0)]
    tree = KDTree([])
    built: List[Point] = []
    for p in points:
        tree.insert(p)
        built.append(p)
        # Verify nearest for a random query
        query = (random.uniform(-5, 5), random.uniform(-5, 5))
        kd_point, kd_dist = tree.nearest(query)
        bf_point, bf_dist = _brute_force_nearest(built, query)
        assert kd_point == bf_point
        assert math.isclose(kd_dist, bf_dist, rel_tol=1e-9)


if __name__ == "__main__":
    _run_basic_tests()
    _run_randomized_tests()
    _run_insertion_test()
    print("All KD‑Tree unit tests passed.")
