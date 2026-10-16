from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, Iterable

# Type alias for an n-dimensional point or coordinate
Point = Tuple[float, ...]
# Bounding box represented by lower and upper coordinate tuples
BBox = Tuple[Point, Point]

def bbox_area(bbox: BBox) -> float:
    """Compute hypervolume of a bounding box."""
    lower, upper = bbox
    area = 1.0
    for l, u in zip(lower, upper):
        area *= max(0.0, u - l)
    return area

def bbox_intersects(a: BBox, b: BBox) -> bool:
    """Return True if bounding boxes a and b intersect."""
    for al, au, bl, bu in zip(a[0], a[1], b[0], b[1]):
        if au < bl or bu < al:
            return False
    return True

def bbox_contains(a: BBox, b: BBox) -> bool:
    """Return True if a completely contains b."""
    for al, au, bl, bu in zip(a[0], a[1], b[0], b[1]):
        if al > bl or au < bu:
            return False
    return True

def bbox_merge(a: BBox, b: BBox) -> BBox:
    """Return the minimal bounding box that contains both a and b."""
    lower = tuple(min(al, bl) for al, bl in zip(a[0], b[0]))
    upper = tuple(max(au, bu) for au, bu in zip(a[1], b[1]))
    return (lower, upper)

def bbox_enlargement(a: BBox, b: BBox) -> float:
    """Return the area increase needed to merge a and b."""
    return bbox_area(bbox_merge(a, b)) - bbox_area(a)

@dataclass
class Entry:
    bbox: BBox
    child: Optional["Node"] = None
    data: Any = None

@dataclass
class Node:
    entries: List[Entry] = field(default_factory=list)
    leaf: bool = True

    def bbox(self) -> BBox:
        """Compute the bounding box of all entries."""
        if not self.entries:
            raise ValueError("Empty node has no bounding box")
        lower = list(self.entries[0].bbox[0])
        upper = list(self.entries[0].bbox[1])
        for e in self.entries[1:]:
            lower = [min(l, el) for l, el in zip(lower, e.bbox[0])]
            upper = [max(u, eu) for u, eu in zip(upper, e.bbox[1])]
        return (tuple(lower), tuple(upper))

class RTree:
    def __init__(self, max_entries: int = 4, min_entries: int = 2):
        if max_entries < 3:
            raise ValueError("max_entries must be at least 3")
        if min_entries < 1 or min_entries > max_entries // 2:
            raise ValueError("min_entries out of bounds")
        self.max_entries = max_entries
        self.min_entries = min_entries
        self.root = Node(leaf=True)

    def insert(self, data: Any, bbox: BBox) -> None:
        leaf = self._choose_leaf(self.root, bbox)
        leaf.entries.append(Entry(bbox=bbox, data=data))
        if len(leaf.entries) > self.max_entries:
            self._split_node(leaf)
        self._adjust_tree(leaf)

    def search(self, query: BBox) -> List[Any]:
        result: List[Any] = []
        self._search(self.root, query, result)
        return result

    def _search(self, node: Node, query: BBox, result: List[Any]) -> None:
        for e in node.entries:
            if bbox_intersects(e.bbox, query):
                if node.leaf:
                    result.append(e.data)
                else:
                    self._search(e.child, query, result)

    def _choose_leaf(self, node: Node, bbox: BBox) -> Node:
        if node.leaf:
            return node
        # Choose subtree that requires least enlargement
        best = None
        best_enlargement = None
        best_area = None
        for e in node.entries:
            enlargement = bbox_enlargement(e.bbox, bbox)
            area = bbox_area(e.bbox)
            if best is None or enlargement < best_enlargement or (enlargement == best_enlargement and area < best_area):
                best = e
                best_enlargement = enlargement
                best_area = area
        return self._choose_leaf(best.child, bbox)

    def _adjust_tree(self, node: Node) -> None:
        while node is not None:
            if node is self.root:
                if len(node.entries) > self.max_entries:
                    self._split_root()
                break
            parent = self._find_parent(self.root, node)
            if parent is None:
                break
            # Update parent's entry bbox
            for e in parent.entries:
                if e.child is node:
                    e.bbox = node.bbox()
                    break
            if len(node.entries) > self.max_entries:
                self._split_node(node)
            node = parent

    def _split_root(self) -> None:
        old_root = self.root
        self.root = Node(leaf=False)
        left, right = self._split_entries(old_root.entries)
        self.root.entries.append(Entry(bbox=left.bbox(), child=left))
        self.root.entries.append(Entry(bbox=right.bbox(), child=right))

    def _split_node(self, node: Node) -> None:
        left, right = self._split_entries(node.entries)
        node.entries = left.entries
        node.leaf = left.leaf
        new_node = Node(entries=right.entries, leaf=right.leaf)
        parent = self._find_parent(self.root, node)
        if parent is None:
            # node is root
            self.root = Node(leaf=False)
            self.root.entries.append(Entry(bbox=node.bbox(), child=node))
            self.root.entries.append(Entry(bbox=new_node.bbox(), child=new_node))
        else:
            parent.entries.append(Entry(bbox=new_node.bbox(), child=new_node))

    def _split_entries(self, entries: List[Entry]) -> Tuple[Node, Node]:
        # Quadratic split
        seed1, seed2 = self._pick_seeds(entries)
        group1 = Node(entries=[seed1], leaf=seed1.child is None)
        group2 = Node(entries=[seed2], leaf=seed2.child is None)
        remaining = [e for e in entries if e not in (seed1, seed2)]
        while remaining:
            if len(group1.entries) + len(remaining) == self.min_entries:
                group1.entries.extend(remaining)
                break
            if len(group2.entries) + len(remaining) == self.min_entries:
                group2.entries.extend(remaining)
                break
            e = self._pick_next(group1, group2, remaining)
            remaining.remove(e)
            enlargement1 = bbox_enlargement(group1.bbox(), e.bbox)
            enlargement2 = bbox_enlargement(group2.bbox(), e.bbox)
            if enlargement1 < enlargement2:
                group1.entries.append(e)
            elif enlargement2 < enlargement1:
                group2.entries.append(e)
            else:
                if bbox_area(group1.bbox()) < bbox_area(group2.bbox()):
                    group1.entries.append(e)
                else:
                    group2.entries.append(e)
        return group1, group2

    def _pick_seeds(self, entries: List[Entry]) -> Tuple[Entry, Entry]:
        max_d = -1.0
        seed1 = seed2 = None
        for i, e1 in enumerate(entries):
            for e2 in entries[i+1:]:
                d = bbox_enlargement(e1.bbox, e2.bbox)
                if d > max_d:
                    max_d = d
                    seed1, seed2 = e1, e2
        return seed1, seed2

    def _pick_next(self, g1: Node, g2: Node, remaining: List[Entry]) -> Entry:
        max_diff = -1.0
        chosen = None
        for e in remaining:
            d1 = bbox_enlargement(g1.bbox(), e.bbox)
            d2 = bbox_enlargement(g2.bbox(), e.bbox)
            diff = abs(d1 - d2)
            if diff > max_diff:
                max_diff = diff
                chosen = e
        return chosen

    def _find_parent(self, current: Node, child: Node) -> Optional[Node]:
        if current.leaf:
            return None
        for e in current.entries:
            if e.child is child:
                return current
            res = self._find_parent(e.child, child)
            if res:
                return res
        return None
