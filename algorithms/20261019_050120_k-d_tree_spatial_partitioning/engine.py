from __future__ import annotations
from typing import List, Tuple, Optional, Callable
import math

from .types import Point, KDNode


def _distance_sq(p1: Point, p2: Point) -> float:
    """Return squared Euclidean distance between two points."""
    return sum((a - b) ** 2 for a, b in zip(p1, p2))


def _build_kd_tree(points: List[Point], depth: int = 0) -> Optional[KDNode]:
    """Recursively build a balanced k-d tree from a list of points."""
    if not points:
        return None

    k = len(points[0])
    axis = depth % k

    # Sort points and choose median as pivot element
    points.sort(key=lambda point: point[axis])
    median = len(points) // 2

    return KDNode(
        point=points[median],
        left=_build_kd_tree(points[:median], depth + 1),
        right=_build_kd_tree(points[median + 1 :], depth + 1),
        axis=axis,
    )


class KDTree:
    """Balanced k-d tree with nearest‑neighbor query support."""

    __slots__ = ("root", "k")

    def __init__(self, points: List[Point]) -> None:
        if points:
            self.k = len(points[0])
            # Verify all points have the same dimensionality
            for p in points:
                if len(p) != self.k:
                    raise ValueError("All points must have the same dimension")
        else:
            self.k = 0
        # Build a balanced tree; the builder mutates the list, so copy first
        self.root = _build_kd_tree(list(points))

    def nearest(self, target: Point) -> Tuple[Point, float]:
        """Return the nearest point in the tree to *target* and its Euclidean distance."""
        if self.root is None:
            raise ValueError("KDTree is empty")
        if len(target) != self.k:
            raise ValueError("Target dimensionality does not match tree dimensionality")

        best: Tuple[Optional[Point], float] = (None, math.inf)

        def recurse(node: Optional[KDNode]) -> None:
            nonlocal best
            if node is None:
                return

            # Compute distance to current node
            dist_sq = _distance_sq(target, node.point)
            if dist_sq < best[1]:
                best = (node.point, dist_sq)

            axis = node.axis
            diff = target[axis] - node.point[axis]

            # Choose which side to explore first
            first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)

            recurse(first)

            # If the hypersphere crosses the splitting plane, explore the other side
            if diff ** 2 < best[1]:
                recurse(second)

        recurse(self.root)
        assert best[0] is not None  # for type‑checkers
        return best[0], math.sqrt(best[1])

    def insert(self, point: Point) -> None:
        """Insert *point* into the tree, preserving the k‑d property."""
        if self.k == 0:
            # First insertion defines dimensionality
            self.k = len(point)
        elif len(point) != self.k:
            raise ValueError("Inserted point dimensionality mismatch")

        def recurse(node: Optional[KDNode], depth: int) -> KDNode:
            if node is None:
                return KDNode(point=point, axis=depth % self.k)

            axis = node.axis
            if point[axis] < node.point[axis]:
                left = recurse(node.left, depth + 1)
                return KDNode(point=node.point, left=left, right=node.right, axis=axis)
            else:
                right = recurse(node.right, depth + 1)
                return KDNode(point=node.point, left=node.left, right=right, axis=axis)

        self.root = recurse(self.root, 0)
