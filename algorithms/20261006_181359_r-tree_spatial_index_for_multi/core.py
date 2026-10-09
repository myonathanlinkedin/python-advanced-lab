from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
import math

@dataclass
class BBox:
    """Axis-aligned bounding box in 2D space."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def area(self) -> float:
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def intersects(self, other: 'BBox') -> bool:
        return (self.min_x <= other.max_x and self.max_x >= other.min_x and
                self.min_y <= other.max_y and self.max_y >= other.min_y)

    def contains(self, other: 'BBox') -> bool:
        return (self.min_x <= other.min_x and self.max_x >= other.max_x and
                self.min_y <= other.min_y and self.max_y >= other.max_y)

    def union(self, other: 'BBox') -> 'BBox':
        return BBox(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y)
        )

    def distance_to(self, other: 'BBox') -> float:
        dx = max(0.0, max(self.min_x - other.max_x, other.min_x - self.max_x))
        dy = max(0.0, max(self.min_y - other.max_y, other.min_y - self.max_y))
        return math.sqrt(dx * dx + dy * dy)

    def __repr__(self) -> str:
        return f"BBox([{self.min_x},{self.min_y}]-[{self.max_x},{self.max_y}])"


@dataclass
class RTreeNode:
    """Node in the R-Tree."""
    bbox: BBox
    children: List['RTreeNode'] = field(default_factory=list)
    data: Optional[Any] = None  # For leaf nodes
    is_leaf: bool = True

    def update_bbox(self) -> None:
        if self.is_leaf and self.data is not None:
            self.bbox = self.data
        elif self.children:
            combined = self.children[0].bbox
            for child in self.children[1:]:
                combined = combined.union(child.bbox)
            self.bbox = combined
        else:
            self.bbox = BBox(0, 0, 0, 0)


class RTree:
    """R-Tree spatial index for 2D bounding boxes."""

    def __init__(self, max_children: int = 4, min_children: int = 2):
        self.max_children = max_children
        self.min_children = min_children
        self.root = RTreeNode(bbox=BBox(0, 0, 0, 0), is_leaf=True)

    def _choose_subtree(self, bbox: BBox, node: RTreeNode, depth: int = 0) -> RTreeNode:
        if node.is_leaf:
            return node

        best_child = None
        best_enlargement = float('inf')
        best_area = float('inf')

        for child in node.children:
            new_bbox = child.bbox.union(bbox)
            enlargement = new_bbox.area() - child.bbox.area()
            if enlargement < best_enlargement or (enlargement == best_enlargement and child.bbox.area() < best_area):
                best_enlargement = enlargement
                best_area = child.bbox.area()
                best_child = child

        if best_child is None:
            return node
        return self._choose_subtree(bbox, best_child, depth + 1)

    def _insert(self, bbox: BBox, node: RTreeNode) -> None:
        if node.is_leaf:
            node.data = bbox
            node.update_bbox()
            return

        child = self._choose_subtree(bbox, node)
        self._insert(bbox, child)
        node.update_bbox()

        if len(node.children) > self.max_children:
            self._split_node(node)

    def _split_node(self, node: RTreeNode) -> None:
        if node.is_leaf:
            return

        # Simple split: divide children into two groups
        mid = len(node.children) // 2
        left_children = node.children[:mid]
        right_children = node.children[mid:]

        left_node = RTreeNode(bbox=BBox(0, 0, 0, 0), children=left_children, is_leaf=False)
        right_node = RTreeNode(bbox=BBox(0, 0, 0, 0), children=right_children, is_leaf=False)
        left_node.update_bbox()
        right_node.update_bbox()

        node.children = [left_node, right_node]
        node.update_bbox()

    def insert(self, bbox: BBox) -> None:
        """Insert a bounding box into the R-Tree."""
        self._insert(bbox, self.root)

    def _search(self, bbox: BBox, node: RTreeNode, results: List[BBox]) -> None:
        if not node.bbox.intersects(bbox):
            return

        if node.is_leaf:
            if node.data is not None and node.data.intersects(bbox):
                results.append(node.data)
            return

        for child in node.children:
            self._search(bbox, child, results)

    def query(self, bbox: BBox) -> List[BBox]:
        """Query all bounding boxes that intersect with the given bbox."""
        results = []
        self._search(bbox, self.root, results)
        return results

    def nearest_neighbor(self, point: Tuple[float, float]) -> Optional[BBox]:
        """Find the nearest bounding box to a point."""
        best_bbox = None
        best_distance = float('inf')

        def _search_nearest(node: RTreeNode) -> None:
            nonlocal best_bbox, best_distance
            if not node.bbox.intersects(BBox(point[0], point[1], point[0], point[1])):
                # Check if this node's bbox is closer than current best
                dist = node.bbox.distance_to(BBox(point[0], point[1], point[0], point[1]))
                if dist > best_distance:
                    return

            if node.is_leaf:
                if node.data is not None:
                    dist = node.data.distance_to(BBox(point[0], point[1], point[0], point[1]))
                    if dist < best_distance:
                        best_distance = dist
                        best_bbox = node.data
                return

            for child in node.children:
                _search_nearest(child)

        _search_nearest(self.root)
        return best_bbox

    def __len__(self) -> int:
        count = 0

        def _count(node: RTreeNode) -> None:
            nonlocal count
            if node.is_leaf and node.data is not None:
                count += 1
            else:
                for child in node.children:
                    _count(child)

        _count(self.root)
        return count

    def __repr__(self) -> str:
        return f"RTree(size={len(self)}, max_children={self.max_children})"
