from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Iterable, Optional

BoundingBox = Tuple[Tuple[float, ...], Tuple[float, ...]]  # ((min1, min2, ...), (max1, max2, ...))

def combine_boxes(a: BoundingBox, b: BoundingBox) -> BoundingBox:
    """Return the minimal bounding box that contains both a and b."""
    min_a, max_a = a
    min_b, max_b = b
    return (
        tuple(min(ma, mb) for ma, mb in zip(min_a, min_b)),
        tuple(max(ma, mb) for ma, mb in zip(max_a, max_b)),
    )

def box_overlaps(a: BoundingBox, b: BoundingBox) -> bool:
    """Return True if boxes a and b overlap."""
    min_a, max_a = a
    min_b, max_b = b
    return all(ma >= mb_min and mb >= ma_min for ma_min, ma, mb_min, mb in zip(min_a, max_a, min_b, max_b))

def box_contains(a: BoundingBox, b: BoundingBox) -> bool:
    """Return True if box a completely contains box b."""
    min_a, max_a = a
    min_b, max_b = b
    return all(ma <= mb_min and mb <= ma_max for ma_min, ma, mb_min, mb in zip(min_a, max_a, min_b, max_b))

@dataclass
class Entry:
    bbox: BoundingBox
    child: Optional["Node"] = None
    data: Any = None

@dataclass
class Node:
    is_leaf: bool
    entries: List[Entry] = field(default_factory=list)

    @property
    def bbox(self) -> BoundingBox:
        if not self.entries:
            raise ValueError("Empty node has no bounding box")
        bbox = self.entries[0].bbox
        for e in self.entries[1:]:
            bbox = combine_boxes(bbox, e.bbox)
        return bbox

class RTree:
    def __init__(self, max_entries: int = 4):
        if max_entries < 3:
            raise ValueError("max_entries must be at least 3")
        self.max_entries = max_entries
        self.root = Node(is_leaf=True)

    def insert(self, bbox: BoundingBox, data: Any) -> None:
        leaf = self._choose_leaf(self.root, bbox)
        leaf.entries.append(Entry(bbox=bbox, data=data))
        self._adjust_tree(leaf)

    def search(self, query: BoundingBox) -> List[Any]:
        return self._search(self.root, query)

    def _choose_leaf(self, node: Node, bbox: BoundingBox) -> Node:
        if node.is_leaf:
            return node
        # Choose child that needs least enlargement
        best = None
        best_enlargement = None
        for e in node.entries:
            current = e.bbox
            enlarged = combine_boxes(current, bbox)
            enlargement = self._area(enlarged) - self._area(current)
            if best is None or enlargement < best_enlargement or (enlargement == best_enlargement and self._area(current) < self._area(best.bbox)):
                best = e
                best_enlargement = enlargement
        return self._choose_leaf(best.child, bbox)

    def _adjust_tree(self, node: Node) -> None:
        while node is not None:
            if len(node.entries) > self.max_entries:
                node1, node2 = self._split_node(node)
                if node is self.root:
                    new_root = Node(is_leaf=False, entries=[
                        Entry(bbox=node1.bbox, child=node1),
                        Entry(bbox=node2.bbox, child=node2),
                    ])
                    self.root = new_root
                    return
                else:
                    parent = self._find_parent(self.root, node)
                    # Replace node with two new nodes
                    parent.entries = [e for e in parent.entries if e.child is not node]
                    parent.entries.extend([
                        Entry(bbox=node1.bbox, child=node1),
                        Entry(bbox=node2.bbox, child=node2),
                    ])
                    node = parent
            else:
                node = self._find_parent(self.root, node)

    def _split_node(self, node: Node) -> Tuple[Node, Node]:
        entries = node.entries
        seed1, seed2 = self._pick_seeds(entries)
        group1 = Node(is_leaf=node.is_leaf, entries=[seed1])
        group2 = Node(is_leaf=node.is_leaf, entries=[seed2])
        remaining = [e for e in entries if e not in (seed1, seed2)]
        while remaining:
            if len(group1.entries) + len(remaining) == self.max_entries - self.max_entries // 2:
                group1.entries.extend(remaining)
                break
            if len(group2.entries) + len(remaining) == self.max_entries - self.max_entries // 2:
                group2.entries.extend(remaining)
                break
            e = remaining.pop(0)
            enlargement1 = self._area(combine_boxes(group1.bbox, e.bbox)) - self._area(group1.bbox)
            enlargement2 = self._area(combine_boxes(group2.bbox, e.bbox)) - self._area(group2.bbox)
            if enlargement1 < enlargement2:
                group1.entries.append(e)
            elif enlargement2 < enlargement1:
                group2.entries.append(e)
            else:
                if self._area(group1.bbox) < self._area(group2.bbox):
                    group1.entries.append(e)
                else:
                    group2.entries.append(e)
        return group1, group2

    def _pick_seeds(self, entries: List[Entry]) -> Tuple[Entry, Entry]:
        max_d = -1
        seed1 = seed2 = None
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                e1, e2 = entries[i], entries[j]
                d = self._area(combine_boxes(e1.bbox, e2.bbox)) - self._area(e1.bbox) - self._area(e2.bbox)
                if d > max_d:
                    max_d = d
                    seed1, seed2 = e1, e2
        return seed1, seed2

    def _area(self, bbox: BoundingBox) -> float:
        min_b, max_b = bbox
        prod = 1.0
        for mi, ma in zip(min_b, max_b):
            prod *= max(0.0, ma - mi)
        return prod

    def _find_parent(self, current: Node, child: Node) -> Optional[Node]:
        if current.is_leaf:
            return None
        for e in current.entries:
            if e.child is child:
                return current
            res = self._find_parent(e.child, child)
            if res:
                return res
        return None

    def _search(self, node: Node, query: BoundingBox) -> List[Any]:
        results = []
        for e in node.entries:
            if box_overlaps(e.bbox, query):
                if node.is_leaf:
                    results.append(e.data)
                else:
                    results.extend(self._search(e.child, query))
        return results
