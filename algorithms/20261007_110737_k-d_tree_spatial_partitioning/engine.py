from __future__ import annotations
from typing import List, Optional, Tuple, Sequence
import math

from .types import Point, KDNode


def _choose_axis(points: Sequence[Point]) -> int:
    """Choose axis with greatest spread (max - min)."""
    if not points:
        raise ValueError("Cannot choose axis from empty point list")
    dimensions = len(points[0])
    spreads = []
    for dim in range(dimensions):
        coords = [p[dim] for p in points]
        spreads.append(max(coords) - min(coords))
    return spreads.index(max(spreads))


def build_kdtree(points: List[Point], depth: int = 0) -> Optional[KDNode]:
    """Recursively build a balanced KD‑Tree from a list of points."""
    if not points:
        return None

    k = len(points[0])  # dimensionality
    axis = depth % k

    # Sort point list and choose median as pivot element
    points.sort(key=lambda point: point[axis])
    median = len(points) // 2

    node = KDNode(
        point=points[median],
        axis=axis,
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1 :], depth + 1),
    )
    return node


def _distance_squared(p1: Point, p2: Point) -> float:
    return sum((a - b) ** 2 for a, b in zip(p1, p2))


def nearest_neighbor(root: Optional[KDNode], target: Point) -> Tuple[Point, float]:
    """Return the nearest point in the KD‑Tree to the target and its Euclidean distance."""
    if root is None:
        raise ValueError("KD‑Tree is empty")

    best: Tuple[Point, float] = (root.point, _distance_squared(root.point, target))

    def recurse(node: Optional[KDNode]) -> None:
        nonlocal best
        if node is None:
            return

        point_dist = _distance_squared(node.point, target)
        if point_dist < best[1]:
            best = (node.point, point_dist)

        axis = node.axis
        diff = target[axis] - node.point[axis]

        # Choose which side to explore first
        first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)

        recurse(first)

        # If hypersphere crosses splitting plane, explore the other side
        if diff ** 2 < best[1]:
            recurse(second)

    recurse(root)
    return best[0], math.sqrt(best[1])
