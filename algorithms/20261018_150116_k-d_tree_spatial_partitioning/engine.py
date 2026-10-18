from __future__ import annotations
from typing import List, Tuple, Optional, Callable
import math
from .types import Point, KDNode


def _distance_squared(p1: Point, p2: Point) -> float:
    """Return the squared Euclidean distance between two points."""
    return sum((a - b) ** 2 for a, b in zip(p1, p2))


class KDTree:
    """
    A k-d tree for static point sets with optional dynamic insertion.
    Provides O(log n) average nearest‑neighbor queries in k dimensions.
    """

    def __init__(self, points: List[Point] = None):
        """
        Build a balanced k-d tree from an optional list of points.
        If points is None or empty, an empty tree is created.
        """
        self.k: int = 0
        self.root: Optional[KDNode] = None
        if points:
            if not all(len(p) == len(points[0]) for p in points):
                raise ValueError("All points must have the same dimensionality.")
            self.k = len(points[0])
            self.root = self._build(points, depth=0)

    # --------------------------------------------------------------------- #
    # Tree construction
    # --------------------------------------------------------------------- #
    def _build(self, points: List[Point], depth: int) -> Optional[KDNode]:
        """Recursively build a balanced k-d tree."""
        if not points:
            return None

        axis = depth % self.k
        points.sort(key=lambda pt: pt[axis])
        median = len(points) // 2
        node = KDNode(
            point=points[median],
            axis=axis,
            left=self._build(points[:median], depth + 1),
            right=self._build(points[median + 1 :], depth + 1),
        )
        return node

    # --------------------------------------------------------------------- #
    # Insertion (maintains tree balance only loosely)
    # --------------------------------------------------------------------- #
    def insert(self, point: Point) -> None:
        """Insert a new point into the tree."""
        if self.k == 0:
            # First insertion defines dimensionality.
            self.k = len(point)
        elif len(point) != self.k:
            raise ValueError("Inserted point dimensionality mismatch.")
        self.root = self._insert(self.root, point, depth=0)

    def _insert(self, node: Optional[KDNode], point: Point, depth: int) -> KDNode:
        """Recursive helper for insert."""
        if node is None:
            return KDNode(point=point, axis=depth % self.k)

        axis = node.axis
        if point[axis] < node.point[axis]:
            node.left = self._insert(node.left, point, depth + 1)
        else:
            node.right = self._insert(node.right, point, depth + 1)
        return node

    # --------------------------------------------------------------------- #
    # Nearest‑neighbor search
    # --------------------------------------------------------------------- #
    def nearest(self, query: Point) -> Tuple[Point, float]:
        """
        Return the nearest point in the tree to `query` and the Euclidean distance.
        Raises ValueError if the tree is empty.
        """
        if self.root is None:
            raise ValueError("Cannot query an empty KDTree.")
        if len(query) != self.k:
            raise ValueError("Query point dimensionality mismatch.")
        best_point, best_dist_sq = self._nn_search(self.root, query, float("inf"), None)
        return best_point, math.sqrt(best_dist_sq)

    def _nn_search(
        self,
        node: Optional[KDNode],
        query: Point,
        best_dist_sq: float,
        best_point: Optional[Point],
    ) -> Tuple[Point, float]:
        """Recursive nearest‑neighbor search."""
        if node is None:
            return best_point, best_dist_sq

        # Compute distance to current node.
        node_dist_sq = _distance_squared(query, node.point)
        if node_dist_sq < best_dist_sq:
            best_dist_sq = node_dist_sq
            best_point = node.point

        axis = node.axis
        diff = query[axis] - node.point[axis]

        # Choose which side to explore first.
        primary, secondary = (node.left, node.right) if diff < 0 else (node.right, node.left)

        # Search the primary side.
        best_point, best_dist_sq = self._nn_search(primary, query, best_dist_sq, best_point)

        # If the hypersphere crosses the splitting plane, search the other side.
        if diff ** 2 < best_dist_sq:
            best_point, best_dist_sq = self._nn_search(secondary, query, best_dist_sq, best_point)

        return best_point, best_dist_sq
