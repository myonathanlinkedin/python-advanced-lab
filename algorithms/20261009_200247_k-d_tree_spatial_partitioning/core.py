from __future__ import annotations
import math
import heapq
from typing import List, Tuple, Optional, Iterable, Any

Point = Tuple[float, ...]


def _distance_sq(p1: Point, p2: Point) -> float:
    """Squared Euclidean distance between two points."""
    return sum((a - b) ** 2 for a, b in zip(p1, p2))


class _KDNode:
    __slots__ = ("point", "left", "right", "axis")

    def __init__(self, point: Point, axis: int,
                 left: Optional['_KDNode'] = None,
                 right: Optional['_KDNode'] = None) -> None:
        self.point: Point = point
        self.axis: int = axis
        self.left: Optional['_KDNode'] = left
        self.right: Optional['_KDNode'] = right

    def __repr__(self) -> str:
        return f"_KDNode(point={self.point}, axis={self.axis})"


class KDTree:
    """
    K-D Tree for k‑dimensional points with exact nearest‑neighbor queries.
    """

    def __init__(self, points: Iterable[Point]) -> None:
        pts = list(points)
        if not pts:
            raise ValueError("KDTree requires at least one point")
        self.k: int = len(pts[0])
        for p in pts:
            if len(p) != self.k:
                raise ValueError("All points must have the same dimension")
        self.root: Optional[_KDNode] = self._build_tree(pts, depth=0)

    def _build_tree(self, points: List[Point], depth: int) -> Optional[_KDNode]:
        if not points:
            return None
        axis = depth % self.k
        points.sort(key=lambda pt: pt[axis])
        median = len(points) // 2
        median_point = points[median]
        left = self._build_tree(points[:median], depth + 1)
        right = self._build_tree(points[median + 1 :], depth + 1)
        return _KDNode(median_point, axis, left, right)

    def nearest(self, query: Point, k: int = 1) -> List[Tuple[Point, float]]:
        """
        Return the k nearest points to *query* as (point, distance) tuples.
        Distance is Euclidean (not squared).
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if len(query) != self.k:
            raise ValueError("Query point dimensionality mismatch")
        heap: List[Tuple[float, Point]] = []  # max‑heap via negative distance

        def _search(node: Optional[_KDNode]) -> None:
            if node is None:
                return
            dist_sq = _distance_sq(query, node.point)
            # Use negative distance to simulate max‑heap with heapq (which is min‑heap)
            if len(heap) < k:
                heapq.heappush(heap, (-dist_sq, node.point))
            else:
                if dist_sq < -heap[0][0]:
                    heapq.heapreplace(heap, (-dist_sq, node.point))

            axis = node.axis
            diff = query[axis] - node.point[axis]
            close, away = (node.left, node.right) if diff <= 0 else (node.right, node.left)

            _search(close)

            # Check whether we need to explore the away branch
            if len(heap) < k or diff ** 2 < -heap[0][0]:
                _search(away)

        _search(self.root)

        # Convert to (point, distance) sorted by distance ascending
        result = [ (pt, math.sqrt(-d_sq)) for d_sq, pt in heap ]
        result.sort(key=lambda pair: pair[1])
        return result

    def __repr__(self) -> str:
        return f"KDTree(k={self.k})"


__all__ = ["KDTree", "Point"]
