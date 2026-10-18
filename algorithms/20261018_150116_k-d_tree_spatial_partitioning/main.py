from __future__ import annotations
import random
import math
from typing import List, Tuple
from custom_types import Point
from engine import KDTree

def _brute_force_nearest(points: List[Point], query: Point) -> Tuple[Point, float]:
    """Reference implementation using linear scan."""
    best = points[0]
    best_dist_sq = sum((a - b) ** 2 for a, b in zip(best, query))
    for pt in points[1:]:
        d_sq = sum((a - b) ** 2 for a, b in zip(pt, query))
        if d_sq < best_dist_sq:
            best = pt
            best_dist_sq = d_sq
    return best, math.sqrt(best_dist_sq)


def _test_basic_functionality() -> None:
    # Simple 2‑D case.
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]
    tree = KDTree(points)

    # Query exactly at a stored point.
    q = (1.0, 1.0)
    nn, dist = tree.nearest(q)
    assert nn == q and math.isclose(dist, 0.0), "Exact match failed."

    # Query between points.
    q = (3.0, 3.0)
    nn, dist = tree.nearest(q)
    assert nn == (2.0, 2.0), "Nearest neighbor incorrect for 2‑D case."

    # Insert a new point and query again.
    tree.insert((3.1, 3.1))
    nn, dist = tree.nearest(q)
    assert nn == (3.1, 3.1), "Insertion or nearest search failed after insert."

def _test_random_high_dimensional() -> None:
    random.seed(42)
    k = 5  # dimensionality
    n = 200  # number of points
    points = [
        tuple(random.uniform(-100, 100) for _ in range(k))
        for _ in range(n)
    ]
    tree = KDTree(points)

    # Perform 50 random queries and compare with brute force.
    for _ in range(50):
        query = tuple(random.uniform(-100, 100) for _ in range(k))
        nn_tree, dist_tree = tree.nearest(query)
        nn_brute, dist_brute = _brute_force_nearest(points, query)
        assert nn_tree == nn_brute, "Nearest point mismatch."
        assert math.isclose(dist_tree, dist_brute, rel_tol=1e-9), "Distance mismatch."

def _test_edge_cases() -> None:
    # Empty tree should raise.
    empty = KDTree()
    try:
        empty.nearest((0.0,))
        assert False, "Empty tree did not raise."
    except ValueError:
        pass

    # Single‑point tree.
    single = KDTree([(7.5, -3.2, 4.1)])
    nn, dist = single.nearest((0.0, 0.0, 0.0))
    expected_dist = math.sqrt((7.5)**2 + (-3.2)**2 + (4.1)**2)
    assert nn == (7.5, -3.2, 4.1) and math.isclose(dist, expected_dist), "Single point case failed."

    # Duplicate points.
    dup_points = [(1.0, 1.0), (1.0, 1.0), (2.0, 2.0)]
    dup_tree = KDTree(dup_points)
    nn, dist = dup_tree.nearest((1.0, 1.0))
    assert nn == (1.0, 1.0) and math.isclose(dist, 0.0), "Duplicate handling failed."

def _run_all_tests() -> None:
    _test_basic_functionality()
    _test_random_high_dimensional()
    _test_edge_cases()
    print("All KDTree unit tests passed.")

if __name__ == "__main__":
    _run_all_tests()
    # Demonstration
    demo_points = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    tree = KDTree(demo_points)
    query = (9, 2)
    nearest_point, distance = tree.nearest(query)
    print(f"Query point: {query}")
    print(f"Nearest neighbor: {nearest_point} (distance = {distance:.4f})")
