from __future__ import annotations
from typing import List, Tuple, Any, Optional, Iterable
import math

Point = Tuple[float, ...]
Box = Tuple[Point, Point]  # (min_coords, max_coords)


def box_area(box: Box) -> float:
    min_pt, max_pt = box
    return math.prod(max_pt[i] - min_pt[i] for i in range(len(min_pt)))


def combine_boxes(a: Box, b: Box) -> Box:
    min_a, max_a = a
    min_b, max_b = b
    return tuple(min(min_a[i], min_b[i]) for i in range(len(min_a))), \
           tuple(max(max_a[i], max_b[i]) for i in range(len(max_a)))


def box_enlargement(original: Box, added: Box) -> float:
    combined = combine_boxes(original, added)
    return box_area(combined) - box_area(original)


def boxes_intersect(a: Box, b: Box) -> bool:
    min_a, max_a = a
    min_b, max_b = b
    return all(min_a[i] <= max_b[i] and max_a[i] >= min_b[i] for i in range(len(min_a)))


class Node:
    __slots__ = ("leaf", "entries", "parent", "box", "max_entries", "min_entries")

    def __init__(self, leaf: bool, max_entries: int, min_entries: int, parent: Optional[Node] = None):
        self.leaf: bool = leaf
        self.entries: List[Tuple[Box, Any]] = []  # (box, child) for internal, (box, obj) for leaf
        self.parent: Optional[Node] = parent
        self.box: Optional[Box] = None
        self.max_entries = max_entries
        self.min_entries = min_entries

    def update_box(self) -> None:
        if not self.entries:
            self.box = None
            return
        cur = self.entries[0][0]
        for b, _ in self.entries[1:]:
            cur = combine_boxes(cur, b)
        self.box = cur

    def is_overflow(self) -> bool:
        return len(self.entries) > self.max_entries

    def add_entry(self, box: Box, obj: Any) -> None:
        self.entries.append((box, obj))
        if self.box is None:
            self.box = box
        else:
            self.box = combine_boxes(self.box, box)


def linear_split(node: Node) -> Tuple[Node, Node]:
    # Choose seeds
    dim = len(node.box[0])
    highest_low = [(-math.inf, None)] * dim
    lowest_high = [(math.inf, None)] * dim
    for entry in node.entries:
        b = entry[0]
        for d in range(dim):
            low = b[0][d]
            high = b[1][d]
            if low > highest_low[d][0]:
                highest_low[d] = (low, entry)
            if high < lowest_high[d][0]:
                lowest_high[d] = (high, entry)

    best_d = -1
    max_norm = -1.0
    for d in range(dim):
        low_val, _ = highest_low[d]
        high_val, _ = lowest_high[d]
        width = node.box[1][d] - node.box[0][d]
        if width == 0:
            norm = 0
        else:
            norm = (low_val - high_val) / width
        if norm > max_norm:
            max_norm = norm
            best_d = d

    seed1 = highest_low[best_d][1]
    seed2 = lowest_high[best_d][1]
    group1 = Node(node.leaf, node.max_entries, node.min_entries, node.parent)
    group2 = Node(node.leaf, node.max_entries, node.min_entries, node.parent)
    group1.add_entry(*seed1)
    group2.add_entry(*seed2)

    remaining = [e for e in node.entries if e not in (seed1, seed2)]

    while remaining:
        # enforce min fill
        if len(group1.entries) + len(remaining) == node.min_entries:
            for e in remaining:
                group1.add_entry(*e)
            break
        if len(group2.entries) + len(remaining) == node.min_entries:
            for e in remaining:
                group2.add_entry(*e)
            break

        # pick next entry
        diff = -math.inf
        chosen = None
        target_group = None
        for e in remaining:
            d1 = box_enlargement(group1.box, e[0])
            d2 = box_enlargement(group2.box, e[0])
            delta = abs(d1 - d2)
            if delta > diff:
                diff = delta
                chosen = e
                target_group = group1 if d1 < d2 else group2
        target_group.add_entry(*chosen)
        remaining.remove(chosen)

    return group1, group2


class RTree:
    def __init__(self, dimension: int, max_entries: int = 8):
        self.dimension = dimension
        self.max_entries = max_entries
        self.min_entries = max(2, max_entries // 2)
        self.root = Node(leaf=True, max_entries=self.max_entries, min_entries=self.min_entries)

    def insert(self, box: Box, obj: Any) -> None:
        leaf = self._choose_leaf(self.root, box)
        leaf.add_entry(box, obj)
        self._adjust_tree(leaf)

    def _choose_leaf(self, node: Node, box: Box) -> Node:
        if node.leaf:
            return node
        # Choose the child requiring least enlargement
        best_child = None
        min_enlargement = math.inf
        min_area = math.inf
        for child_box, child in node.entries:
            enlargement = box_enlargement(child_box, box)
            area = box_area(child_box)
            if enlargement < min_enlargement or (enlargement == min_enlargement and area < min_area):
                min_enlargement = enlargement
                min_area = area
                best_child = child
        return self._choose_leaf(best_child, box)

    def _adjust_tree(self, node: Node) -> None:
        while node is not None:
            if node.is_overflow():
                n1, n2 = linear_split(node)
                if node.parent is None:
                    # create new root
                    new_root = Node(leaf=False, max_entries=self.max_entries, min_entries=self.min_entries)
                    n1.parent = new_root
                    n2.parent = new_root
                    new_root.add_entry(n1.box, n1)
                    new_root.add_entry(n2.box, n2)
                    self.root = new_root
                    return
                else:
                    # replace node with n1, add n2 to parent
                    parent = node.parent
                    # remove old entry
                    parent.entries = [(b, c) for b, c in parent.entries if c is not node]
                    n1.parent = parent
                    parent.add_entry(n1.box, n1)
                    n2.parent = parent
                    parent.add_entry(n2.box, n2)
                    node = parent
            else:
                node.update_box()
                node = node.parent

    def search(self, box: Box) -> List[Any]:
        result: List[Any] = []
        self._search_recursive(self.root, box, result)
        return result

    def _search_recursive(self, node: Node, box: Box, result: List[Any]) -> None:
        if node.leaf:
            for entry_box, obj in node.entries:
                if boxes_intersect(entry_box, box):
                    result.append(obj)
        else:
            for child_box, child in node.entries:
                if boxes_intersect(child_box, box):
                    self._search_recursive(child, box, result)
