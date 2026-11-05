from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Any

Point = Tuple[float, ...]


@dataclass
class _Node:
    point: Point
    left: Optional[_Node] = None
    right: Optional[_Node] = None
    axis: int = 0


class KDTree:
    """
    A simple k-d tree implementation for Euclidean space.
    Supports construction from a list of points, insertion, and nearest neighbor search.
    """

    __slots__ = ("root", "k", "_size")

    def __init__(self, points: Optional[Iterable[Point]] = None) -> None:
        self.root: Optional[_Node] = None
        self.k: int = 0
        self._size: int = 0
        if points is not None:
            self.build(list(points))

    def build(self, points: List[Point]) -> None:
        if not points:
            self.root = None
            self.k = 0
            self._size = 0
            return
        self.k = len(points[0])
        for p in points:
            if len(p) != self.k:
                raise ValueError("All points must have the same dimension")
        self.root = self._build_recursive(points, depth=0)
        self._size = len(points)

    def _build_recursive(self, points: List[Point], depth: int) -> Optional[_Node]:
        if not points:
            return None
        axis = depth % self.k
        points.sort(key=lambda point: point[axis])
        median = len(points) // 2
        node = _Node(
            point=points[median],
            left=self._build_recursive(points[:median], depth + 1),
            right=self._build_recursive(points[median + 1 :], depth + 1),
            axis=axis,
        )
        return node

    def insert(self, point: Point) -> None:
        if not point:
            raise ValueError("Point must not be empty")
        if self.k == 0:
            self.k = len(point)
        elif len(point) != self.k:
            raise ValueError("Point dimension does not match tree dimension")
        self.root = self._insert_recursive(self.root, point, depth=0)
        self._size += 1

    def _insert_recursive(
        self, node: Optional[_Node], point: Point, depth: int
    ) -> _Node:
        if node is None:
            return _Node(point=point, axis=depth % self.k)
        axis = node.axis
        if point[axis] < node.point[axis]:
            node.left = self._insert_recursive(node.left, point, depth + 1)
        else:
            node.right = self._insert_recursive(node.right, point, depth + 1)
        return node

    def nearest_neighbor(self, query: Point) -> Optional[Point]:
        if self.root is None:
            return None
        if len(query) != self.k:
            raise ValueError("Query dimension does not match tree dimension")
        best = [None, float("inf")]  # [best_point, best_dist_sq]
        self._nn_recursive(self.root, query, best)
        return best[0]

    def _nn_recursive(
        self, node: Optional[_Node], query: Point, best: List[Any]
    ) -> None:
        if node is None:
            return
        dist_sq = self._squared_distance(query, node.point)
        if dist_sq < best[1]:
            best[0], best[1] = node.point, dist_sq
        axis = node.axis
        diff = query[axis] - node.point[axis]
        close, away = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._nn_recursive(close, query, best)
        if diff * diff < best[1]:
            self._nn_recursive(away, query, best)

    @staticmethod
    def _squared_distance(a: Point, b: Point) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def __len__(self) -> int:
        return self._size

    def __iter__(self):
        stack: List[_Node] = []
        node = self.root
        while stack or node:
            if node:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                yield node.point
                node = node.right

    def __contains__(self, point: Point) -> bool:
        return self._contains_recursive(self.root, point)

    def _contains_recursive(self, node: Optional[_Node], point: Point) -> bool:
        if node is None:
            return False
        if node.point == point:
            return True
        axis = node.axis
        if point[axis] < node.point[axis]:
            return self._contains_recursive(node.left, point)
        return self._contains_recursive(node.right, point)

    def __repr__(self) -> str:
        points = list(self)
        return f"KDTree(k={self.k}, points={points})"
