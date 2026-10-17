from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Any
import math

Point = Tuple[float, ...]


def _distance_sq(p1: Point, p2: Point) -> float:
    """Return squared Euclidean distance between two points."""
    return sum((a - b) ** 2 for a, b in zip(p1, p2))


@dataclass
class _KDNode:
    point: Point
    left: Optional['_KDNode']
    right: Optional['_KDNode']
    axis: int


class KDTree:
    """
    K‑D Tree for k‑dimensional points with exact nearest‑neighbor queries.
    The tree is immutable after construction.
    """

    def __init__(self, points: List[Point]) -> None:
        if not points:
            raise ValueError("KDTree requires at least one point")
        self.k = len(points[0])
        for p in points:
            if len(p) != self.k:
                raise ValueError("All points must have the same dimension")
        self.root: Optional[_KDNode] = self._build(points, depth=0)

    def _build(self, points: List[Point], depth: int) -> Optional[_KDNode]:
        """Recursively build a balanced KD‑Tree."""
        if not points:
            return None
        axis = depth % self.k
        points.sort(key=lambda pt: pt[axis])
        median = len(points) // 2
        return _KDNode(
            point=points[median],
            left=self._build(points[:median], depth + 1),
            right=self._build(points[median + 1 :], depth + 1),
            axis=axis,
        )

    def nearest(self, target: Point) -> Tuple[Point, float]:
        """
        Return the nearest point in the tree to *target* and its Euclidean distance.
        Raises ValueError if *target* dimension differs from the tree's dimension.
        """
        if len(target) != self.k:
            raise ValueError("Target point dimension mismatch")
        best: Tuple[float, Point] = (math.inf, target)  # (dist_sq, point)

        def recurse(node: Optional[_KDNode]) -> None:
            nonlocal best
            if node is None:
                return
            # Compute distance to current node
            dist_sq = _distance_sq(target, node.point)
            if dist_sq < best[0]:
                best = (dist_sq, node.point)

            axis = node.axis
            diff = target[axis] - node.point[axis]

            # Choose side to explore first
            first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)

            recurse(first)

            # Prune if hyperplane distance exceeds current best
            if diff ** 2 < best[0]:
                recurse(second)

        recurse(self.root)
        nearest_point = best[1]
        distance = math.sqrt(best[0])
        return nearest_point, distance

    def __len__(self) -> int:
        """Return number of points stored in the tree."""
        def count(node: Optional[_KDNode]) -> int:
            if node is None:
                return 0
            return 1 + count(node.left) + count(node.right)

        return count(self.root)


__all__ = ["KDTree", "Point"]
