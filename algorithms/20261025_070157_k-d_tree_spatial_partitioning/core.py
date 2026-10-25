from __future__ import annotations
import math
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

Point = Tuple[float, ...]


def euclidean_distance(p1: Point, p2: Point) -> float:
    """Return Euclidean distance between two points of equal dimension."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


@dataclass(frozen=True)
class KDNode:
    point: Point
    left: Optional["KDNode"]
    right: Optional["KDNode"]
    axis: int

    def __repr__(self) -> str:
        return f"KDNode(point={self.point}, axis={self.axis})"


class KDTree:
    """Static k‑d tree supporting exact k‑nearest‑neighbor queries."""

    def __init__(self, points: Iterable[Point]) -> None:
        points = list(points)
        if not points:
            raise ValueError("KDTree requires at least one point")
        self.k = len(points[0])
        for p in points:
            if len(p) != self.k:
                raise ValueError("All points must have the same dimension")
        self.root = self._build(points, depth=0)

    def _build(self, points: List[Point], depth: int) -> Optional[KDNode]:
        if not points:
            return None
        axis = depth % self.k
        points.sort(key=lambda pt: pt[axis])
        median = len(points) // 2
        # Deterministic tie‑break: left side gets earlier points when equal
        return KDNode(
            point=points[median],
            left=self._build(points[:median], depth + 1),
            right=self._build(points[median + 1 :], depth + 1),
            axis=axis,
        )

    def nearest(self, query: Point, k: int = 1) -> List[Tuple[float, Point]]:
        """
        Return a list of the k nearest neighbours to ``query``.
        Each entry is a tuple (distance, point) sorted by increasing distance.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if len(query) != self.k:
            raise ValueError("Query point dimension mismatch")
        heap: List[Tuple[float, Point]] = []  # max‑heap via negative distance

        def push(candidate: Point) -> None:
            dist = euclidean_distance(query, candidate)
            if len(heap) < k:
                heapq.heappush(heap, (-dist, candidate))
            else:
                if dist < -heap[0][0]:
                    heapq.heapreplace(heap, (-dist, candidate))

        def recurse(node: Optional[KDNode]) -> None:
            if node is None:
                return
            push(node.point)
            axis = node.axis
            diff = query[axis] - node.point[axis]
            # Choose side that query lies in first
            near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
            recurse(near)
            # If hypersphere crosses splitting plane, explore far side
            if len(heap) < k or abs(diff) < -heap[0][0]:
                recurse(far)

        recurse(self.root)
        # Convert max‑heap to sorted list (ascending distance)
        result = sorted(((-d, pt) for d, pt in heap), key=lambda x: x[0])
        return result

    def __repr__(self) -> str:
        return f"KDTree(k={self.k})"


__all__ = ["KDTree", "euclidean_distance", "Point"]
