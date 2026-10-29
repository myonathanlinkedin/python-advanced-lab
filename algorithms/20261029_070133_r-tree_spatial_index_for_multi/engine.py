from __future__ import annotations
from typing import List, Any, Tuple, Iterable, Optional
from custom_types import BoundingBox, RTreeEntry, RTreeNode

class RTree:
    """A simple R-Tree implementation with quadratic split."""
    MAX_ENTRIES: int = 4
    MIN_ENTRIES: int = 2

    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self.root = RTreeNode(is_leaf=True)

    def insert(self, bbox: BoundingBox, data: Any) -> None:
        if len(bbox.mins) != self.dimensions:
            raise ValueError("Bounding box dimensionality mismatch")
        leaf = self._choose_leaf(self.root, bbox)
        leaf.entries.append(RTreeEntry(bbox=bbox, data=data))
        if len(leaf.entries) > self.MAX_ENTRIES:
            self._split_node(leaf)

    def search(self, query: BoundingBox) -> List[Any]:
        if len(query.mins) != self.dimensions:
            raise ValueError("Query dimensionality mismatch")
        results: List[Any] = []
        self._search_recursive(self.root, query, results)
        return results

    # Internal helper methods
    def _choose_leaf(self, node: RTreeNode, bbox: BoundingBox) -> RTreeNode:
        if node.is_leaf:
            return node
        # Choose child that requires least enlargement
        best_entry = None
        best_increase = None
        best_area = None
        for entry in node.entries:
            current_area = entry.bbox.area()
            enlarged_bbox = entry.bbox.union(bbox)
            enlarged_area = enlarged_bbox.area()
            increase = enlarged_area - current_area
            if best_increase is None or increase < best_increase or (increase == best_increase and current_area < best_area):
                best_increase = increase
                best_area = current_area
                best_entry = entry
        return self._choose_leaf(best_entry.child, bbox)

    def _search_recursive(self, node: RTreeNode, query: BoundingBox, results: List[Any]) -> None:
        for entry in node.entries:
            if entry.bbox.intersects(query):
                if node.is_leaf:
                    results.append(entry.data)
                else:
                    self._search_recursive(entry.child, query, results)

    def _split_node(self, node: RTreeNode) -> None:
        group1, group2 = self._quadratic_split(node.entries)
        node.entries = group1
        if node is self.root:
            # Create new root
            new_root = RTreeNode(is_leaf=False)
            new_root.entries.append(RTreeEntry(bbox=node.bbox(), child=node))
            new_root.entries.append(RTreeEntry(bbox=self._bbox_of_entries(group2), child=RTreeNode(is_leaf=node.is_leaf, entries=group2)))
            self.root = new_root
        else:
            # Replace node's entries with group1 and add group2 as sibling
            sibling = RTreeNode(is_leaf=node.is_leaf, entries=group2)
            parent = self._find_parent(self.root, node)
            if parent is None:
                raise RuntimeError("Parent not found during split")
            parent.entries.append(RTreeEntry(bbox=sibling.bbox(), child=sibling))

    def _quadratic_split(self, entries: List[RTreeEntry]) -> Tuple[List[RTreeEntry], List[RTreeEntry]]:
        # Pick seeds
        max_d = None
        seed1 = seed2 = None
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                b1 = entries[i].bbox
                b2 = entries[j].bbox
                d = b1.union(b2).area() - b1.area() - b2.area()
                if max_d is None or d > max_d:
                    max_d = d
                    seed1, seed2 = entries[i], entries[j]
        group1 = [seed1]
        group2 = [seed2]
        remaining = [e for e in entries if e not in (seed1, seed2)]

        while remaining:
            if (len(group1) + len(remaining)) == self.MIN_ENTRIES:
                group1.extend(remaining)
                break
            if (len(group2) + len(remaining)) == self.MIN_ENTRIES:
                group2.extend(remaining)
                break
            # Pick next entry
            next_entry = None
            max_diff = None
            for entry in remaining:
                d1 = self._enlargement_needed(group1, entry)
                d2 = self._enlargement_needed(group2, entry)
                diff = abs(d1 - d2)
                if max_diff is None or diff > max_diff:
                    max_diff = diff
                    next_entry = entry
            # Assign to group with smaller enlargement
            d1 = self._enlargement_needed(group1, next_entry)
            d2 = self._enlargement_needed(group2, next_entry)
            if d1 < d2 or (d1 == d2 and len(group1) <= len(group2)):
                group1.append(next_entry)
            else:
                group2.append(next_entry)
            remaining.remove(next_entry)
        return group1, group2

    def _enlargement_needed(self, group: List[RTreeEntry], entry: RTreeEntry) -> float:
        if not group:
            return 0.0
        current_bbox = self._bbox_of_entries(group)
        enlarged_bbox = current_bbox.union(entry.bbox)
        return enlarged_bbox.area() - current_bbox.area()

    def _bbox_of_entries(self, entries: List[RTreeEntry]) -> BoundingBox:
        bbox = entries[0].bbox
        for e in entries[1:]:
            bbox = bbox.union(e.bbox)
        return bbox

    def _find_parent(self, current: RTreeNode, child: RTreeNode) -> Optional[RTreeNode]:
        if current.is_leaf:
            return None
        for entry in current.entries:
            if entry.child is child:
                return current
            res = self._find_parent(entry.child, child)
            if res:
                return res
        return None
