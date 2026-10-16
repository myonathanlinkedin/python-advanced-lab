from __future__ import annotations
import random
import math
from typing import List, Tuple

from custom_types import Point, KDNode
from engine import build_kdtree, nearest_neighbor

def _assert_almost_equal(a: float, b: float, eps: float = 1e-9) -> None:
    assert abs(a - b) <= eps, f"{a} !~= {b}"

def test_empty_tree() -> None:
    empty_root = build_kdtree([])
    assert empty_root is None
    try:
        nearest_neighbor(empty_root, (0.0, 0.0))
        assert False, "Expected ValueError for empty tree"
    except ValueError:
        pass

def test_single_point() -> None:
    points = [(1.0, 2.0, 3.0)]
    root = build_kdtree(points)
    nn, dist = nearest_neighbor(root, (5.0, 5.0, 5.0))
    assert nn == points[0]
    expected = math.sqrt(_distance_squared(points[0], (5.0, 5.0, 5.0)))
    _assert_almost_equal(dist, expected)

def test_known_2d() -> None:
    points = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    root = build_kdtree(points.copy())
    query = (9, 2)
    nn, dist = nearest_neighbor(root, query)
    # Manually compute nearest
    expected = min(points, key=lambda p: math.hypot(p[0] - query[0], p[1] - query[1]))
    assert nn == expected
    expected_dist = math.hypot(nn[0] - query[0], nn[1] - query[1])
    _assert_almost_equal(dist, expected_dist)

def test_duplicates() -> None:
    points = [(1, 1), (1, 1), (2, 2), (3, 3)]
    root = build_kdtree(points.copy())
    query = (1, 1)
    nn, dist = nearest_neighbor(root, query)
    assert nn == (1, 1)
    _assert_almost_equal(dist, 0.0)

def test_higher_dimension() -> None:
    points = [
        (0, 0, 0),
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
        (1, 0, 1),
    ]
    root = build_kdtree(points.copy())
    query = (2, 2, 2)
    nn, dist = nearest_neighbor(root, query)
    expected = min(points, key=lambda p: math.sqrt(sum((a - b) ** 2 for a, b in zip(p, query))))
    assert nn == expected
    expected_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(nn, query)))
    _assert_almost_equal(dist, expected_dist)

def _distance_squared(p1: Point, p2: Point) -> float:
    return sum((a - b) ** 2 for a, b in zip(p1, p2))

def demo_random_search() -> None:
    random.seed(42)
    points: List[Point] = [(random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(100)]
    tree = build_kdtree(points)
    query = (random.uniform(-100, 100), random.uniform(-100, 100))
    nn, dist = nearest_neighbor(tree, query)
    # Verify by brute force
    brute_nn = min(points, key=lambda p: math.hypot(p[0] - query[0], p[1] - query[1]))
    brute_dist = math.hypot(brute_nn[0] - query[0], brute_nn[1] - query[1])
    _assert_almost_equal(dist, brute_dist)
    assert nn == brute_nn
    print(f"Query point: {query}")
    print(f"Nearest neighbor (KD‑Tree): {nn} at distance {dist:.4f}")

if __name__ == "__main__":
    test_empty_tree()
    test_single_point()
    test_known_2d()
    test_duplicates()
    test_higher_dimension()
    demo_random_search()
    print("All tests passed.")
