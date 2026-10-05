from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional, List

@dataclass
class KDNode:
    point: Tuple[float, ...]
    left: Optional[KDNode] = None
    right: Optional[KDNode] = None
    axis: int = 0

class KDTree:
    def __init__(self, points: Iterable[Sequence[float]] | None = None) -> None:
        self.root: Optional[KDNode] = None
        self.k: int = 0
        if points:
            pts = [tuple(p) for p in points]
            if not pts:
                return
            self.k = len(pts[0])
            for p in pts:
                if len(p) != self.k:
                    raise ValueError("All points must have the same dimensionality")
            self.root = self._build(pts, depth=0)

    def _build(self, points: List[Tuple[float, ...]], depth: int) -> Optional[KDNode]:
        if not points:
            return None
        axis = depth % self.k
        points.sort(key=lambda x: x[axis])
        median = len(points) // 2
        return KDNode(
            point=points[median],
            left=self._build(points[:median], depth + 1),
            right=self._build(points[median + 1 :], depth + 1),
            axis=axis,
        )

    def insert(self, point: Sequence[float]) -> None:
        pt = tuple(point)
        if not self.root:
            if not pt:
                raise ValueError("Point cannot be empty")
            self.k = len(pt)
            self.root = KDNode(point=pt, axis=0)
            return
        if len(pt) != self.k:
            raise ValueError("Point dimensionality mismatch")
        self.root = self._insert(self.root, pt, depth=0)

    def _insert(self, node: Optional[KDNode], point: Tuple[float, ...], depth: int) -> KDNode:
        if node is None:
            return KDNode(point=point, axis=depth % self.k)
        axis = node.axis
        if point[axis] < node.point[axis]:
            node.left = self._insert(node.left, point, depth + 1)
        else:
            node.right = self._insert(node.right, point, depth + 1)
        return node

    def nearest(self, target: Sequence[float]) -> Tuple[Tuple[float, ...], float]:
        if not self.root:
            raise ValueError("KDTree is empty")
        tgt = tuple(target)
        if len(tgt) != self.k:
            raise ValueError("Target dimensionality mismatch")
        best: List[Optional[Tuple[float, ...]], float] = [None, float("inf")]
        self._nn(self.root, tgt, best)
        return best[0], best[1]

    def _nn(self, node: Optional[KDNode], target: Tuple[float, ...], best: List[Optional[Tuple[float, ...]], float]) -> None:
        if node is None:
            return
        dist_sq = self._squared_distance(node.point, target)
        if dist_sq < best[1]:
            best[0] = node.point
            best[1] = dist_sq
        axis = node.axis
        diff = target[axis] - node.point[axis]
        close, away = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._nn(close, target, best)
        if diff * diff < best[1]:
            self._nn(away, target, best)

    @staticmethod
    def _squared_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

if __name__ == "__main__":
    # Demo
    points = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    tree = KDTree(points)
    query = (9, 2)
    nearest_point, dist_sq = tree.nearest(query)
    print(f"Nearest to {query}: {nearest_point} (dist²={dist_sq})")

    # Unit tests
    def test_kdtree() -> None:
        pts = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
        t = KDTree(pts)
        nn, d2 = t.nearest((9, 2))
        assert nn == (8, 1), f"Expected (8,1), got {nn}"
        assert d2 == 2.0, f"Expected 2.0, got {d2}"

        t.insert((1, 1))
        nn2, d22 = t.nearest((0, 0))
        assert nn2 == (1, 1), f"Expected (1,1), got {nn2}"
        assert d22 == 2.0, f"Expected 2.0, got {d22}"

        empty = KDTree()
        try:
            empty.nearest((0, 0))
            assert False, "Expected ValueError for empty tree"
        except ValueError:
            pass

        single = KDTree([(3, 3)])
        nn3, d33 = single.nearest((3, 3))
        assert nn3 == (3, 3), f"Expected (3,3), got {nn3}"
        assert d33 == 0.0, f"Expected 0.0, got {d33}"

        try:
            t.insert((1, 2, 3))
            assert False, "Expected ValueError for dimensionality mismatch"
        except ValueError:
            pass

        print("All tests passed")

    test_kdtree()
