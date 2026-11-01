from __future__ import annotations
import math
from typing import List, Tuple, Sequence, Optional, Any

Point = Tuple[float, ...]


class KDNode:
    __slots__ = ("point", "left", "right", "axis")

    def __init__(self, point: Point, axis: int,
                 left: Optional[KDNode] = None,
                 right: Optional[KDNode] = None) -> None:
        self.point: Point = point
        self.axis: int = axis
        self.left: Optional[KDNode] = left
        self.right: Optional[KDNode] = right


class KDTree:
    """
    K‑D Tree for static point sets with exact Euclidean nearest‑neighbor queries.
    """

    def __init__(self, points: Sequence[Sequence[float]]) -> None:
        """
        Build a balanced K‑D tree from an iterable of points.
        All points must have the same dimensionality (>0).
        """
        points = [tuple(p) for p in points]
        if not points:
            raise ValueError("KDTree requires at least one point")
        dim = len(points[0])
        if any(len(p) != dim for p in points):
            raise ValueError("All points must have the same dimension")
        self.k: int = dim
        self.root: KDNode = self._build(points, depth=0)

    # --------------------------------------------------------------------- #
    # Construction helpers
    # --------------------------------------------------------------------- #
    def _build(self, pts: List[Point], depth: int) -> Optional[KDNode]:
        if not pts:
            return None
        axis = depth % self.k
        pts.sort(key=lambda p: p[axis])
        median = len(pts) // 2
        node = KDNode(
            point=pts[median],
            axis=axis,
            left=self._build(pts[:median], depth + 1),
            right=self._build(pts[median + 1 :], depth + 1),
        )
        return node

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def nearest(self, target: Sequence[float]) -> Tuple[Point, float]:
        """
        Return the point in the tree closest to *target* and its Euclidean distance.
        """
        if len(target) != self.k:
            raise ValueError("Target dimensionality mismatch")
        best_point, best_dist_sq = self._nn_search(self.root, tuple(target), float("inf"))
        return best_point, math.sqrt(best_dist_sq)

    # --------------------------------------------------------------------- #
    # Search helpers
    # --------------------------------------------------------------------- #
    def _nn_search(
        self,
        node: Optional[KDNode],
        target: Point,
        best_dist_sq: float,
        best_point: Optional[Point] = None,
    ) -> Tuple[Point, float]:
        if node is None:
            assert best_point is not None  # mypy safety
            return best_point, best_dist_sq

        # Compute squared distance from target to current node
        dist_sq = self._squared_distance(node.point, target)
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_point = node.point

        axis = node.axis
        diff = target[axis] - node.point[axis]

        # Choose which side to explore first
        primary, secondary = (node.left, node.right) if diff < 0 else (node.right, node.left)

        # Explore primary branch
        best_point, best_dist_sq = self._nn_search(primary, target, best_dist_sq, best_point)

        # If hyper‑plane could contain a closer point, explore secondary branch
        if diff * diff < best_dist_sq:
            best_point, best_dist_sq = self._nn_search(
                secondary, target, best_dist_sq, best_point
            )

        return best_point, best_dist_sq

    @staticmethod
    def _squared_distance(a: Point, b: Point) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))
