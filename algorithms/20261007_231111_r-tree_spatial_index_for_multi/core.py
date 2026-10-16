from __future__ import annotations
from typing import Any, List, Tuple, Union, Optional
import math

# Type alias for a multidimensional bounding box: (min_coords, max_coords)
BoundingBox = Tuple[Tuple[float, ...], Tuple[float, ...]]

def bbox_union(b1: BoundingBox, b2: BoundingBox) -> BoundingBox:
    """Return the minimal bounding box that contains both b1 and b2."""
    min_coords = tuple(min(a, c) for a, c in zip(b1[0], b2[0]))
    max_coords = tuple(max(b, d) for b, d in zip(b1[1], b2[1]))
    return (min_coords, max_coords)

def bbox_area(b: BoundingBox) -> float:
    """Compute the hypervolume of a bounding box."""
    area = 1.0
    for mn, mx in zip(b[0], b[1]):
        area *= max(0.0, mx - mn)
    return area

def bbox_contains(b: BoundingBox, point: Tuple[float, ...]) -> bool:
    """Check if point lies inside bounding box b."""
    return all(mn <= p <= mx for mn, mx, p in zip(b[0], b[1], point))

def bbox_intersects(b1: BoundingBox, b2: BoundingBox) -> bool:
    """Check if two bounding boxes intersect."""
    return all(a <= d and c <= b for a, b, c, d in zip(b1[0], b1[1], b2[0], b2[1]))

class Node:
    """Internal or leaf node of an R-Tree."""
    def __init__(self, is_leaf: bool):
        self.is_leaf: bool = is_leaf
        self.entries: List[Tuple[BoundingBox, Any]] = []

    def bounding_box(self) -> Optional[BoundingBox]:
        if not self.entries:
            return None
        bbox = self.entries[0][0]
        for entry in self.entries[1:]:
            bbox = bbox_union(bbox, entry[0])
        return bbox

class RTree:
    """Simple R-Tree implementation with quadratic split."""
    def __init__(self, max_entries: int = 4):
        self.max_entries = max_entries
        self.root = Node(is_leaf=True)

    def _choose_leaf(self, node: Node, bbox: BoundingBox) -> Node:
        if node.is_leaf:
            return node
        # Choose child that requires least enlargement
        best_child = None
        best_enlargement = None
        for child_bbox, child_node in node.entries:
            current_area = bbox_area(child_bbox)
            union_bbox = bbox_union(child_bbox, bbox)
            enlargement = bbox_area(union_bbox) - current_area
            if best_enlargement is None or enlargement < best_enlargement:
                best_enlargement = enlargement
                best_child = child_node
        return self._choose_leaf(best_child, bbox)

    def _adjust_tree(self, node: Node, new_node: Optional[Node] = None):
        if node is self.root:
            if new_node:
                # Create new root
                new_root = Node(is_leaf=False)
                new_root.entries.append((node.bounding_box(), node))
                new_root.entries.append((new_node.bounding_box(), new_node))
                self.root = new_root
            return
        parent = self._find_parent(self.root, node)
        # Update bounding box of the entry pointing to node
        for i, (b, child) in enumerate(parent.entries):
            if child is node:
                parent.entries[i] = (node.bounding_box(), node)
                break
        if new_node:
            parent.entries.append((new_node.bounding_box(), new_node))
            if len(parent.entries) > self.max_entries:
                self._split_node(parent)

    def _find_parent(self, current: Node, target: Node) -> Optional[Node]:
        if current.is_leaf:
            return None
        for _, child in current.entries:
            if child is target:
                return current
            res = self._find_parent(child, target)
            if res:
                return res
        return None

    def _split_node(self, node: Node):
        group1, group2 = self._quadratic_split(node.entries)
        node.entries = group1
        new_node = Node(is_leaf=node.is_leaf)
        new_node.entries = group2
        self._adjust_tree(node, new_node)

    def _quadratic_split(self, entries: List[Tuple[BoundingBox, Any]]) -> Tuple[List[Tuple[BoundingBox, Any]], List[Tuple[BoundingBox, Any]]]:
        # Pick seeds
        max_d = -1.0
        seed1 = seed2 = None
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                b1 = entries[i][0]
                b2 = entries[j][0]
                union = bbox_union(b1, b2)
                d = bbox_area(union) - bbox_area(b1) - bbox_area(b2)
                if d > max_d:
                    max_d = d
                    seed1, seed2 = entries[i], entries[j]
        group1 = [seed1]
        group2 = [seed2]
        remaining = [e for e in entries if e not in (seed1, seed2)]
        while remaining:
            if (len(group1) + len(remaining)) == self.max_entries // 2:
                group1.extend(remaining)
                break
            if (len(group2) + len(remaining)) == self.max_entries // 2:
                group2.extend(remaining)
                break
            e = remaining.pop()
            b = e[0]
            inc1 = bbox_area(bbox_union(group1[0][0], b)) - bbox_area(group1[0][0])
            inc2 = bbox_area(bbox_union(group2[0][0], b)) - bbox_area(group2[0][0])
            if inc1 < inc2:
                group1.append(e)
            elif inc2 < inc1:
                group2.append(e)
            else:
                if bbox_area(group1[0][0]) < bbox_area(group2[0][0]):
                    group1.append(e)
                else:
                    group2.append(e)
        return group1, group2

    def insert(self, bbox: BoundingBox, data: Any):
        leaf = self._choose_leaf(self.root, bbox)
        leaf.entries.append((bbox, data))
        if len(leaf.entries) > self.max_entries:
            self._split_node(leaf)

    def search(self, query: BoundingBox) -> List[Any]:
        results: List[Any] = []
        self._search_recursive(self.root, query, results)
        return results

    def _search_recursive(self, node: Node, query: BoundingBox, results: List[Any]):
        for entry_bbox, child in node.entries:
            if bbox_intersects(entry_bbox, query):
                if node.is_leaf:
                    results.append(child)
                else:
                    self._search_recursive(child, query, results)
