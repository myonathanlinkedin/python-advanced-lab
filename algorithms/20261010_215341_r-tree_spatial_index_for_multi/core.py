from __future__ import annotations
from typing import List, Tuple, Any, Optional, Union

Coord = Tuple[float, ...]
Rect = Tuple[Coord, Coord]  # (min_coords, max_coords)


def rect_area(rect: Rect) -> float:
    min_c, max_c = rect
    area = 1.0
    for lo, hi in zip(min_c, max_c):
        area *= max(0.0, hi - lo)
    return area


def rect_union(a: Rect, b: Rect) -> Rect:
    min_a, max_a = a
    min_b, max_b = b
    min_c = tuple(min(x, y) for x, y in zip(min_a, min_b))
    max_c = tuple(max(x, y) for x, y in zip(max_a, max_b))
    return (min_c, max_c)


def rect_intersect(a: Rect, b: Rect) -> bool:
    min_a, max_a = a
    min_b, max_b = b
    for lo_a, hi_a, lo_b, hi_b in zip(min_a, max_a, min_b, max_b):
        if hi_a < lo_b or hi_b < lo_a:
            return False
    return True


def rect_enlargement(rect: Rect, add: Rect) -> float:
    union = rect_union(rect, add)
    return rect_area(union) - rect_area(rect)


class Entry:
    __slots__ = ("rect", "child", "obj")

    def __init__(self, rect: Rect, child: Optional[Node] = None, obj: Any = None):
        self.rect: Rect = rect
        self.child: Optional[Node] = child
        self.obj: Any = obj  # only for leaf entries


class Node:
    __slots__ = ("entries", "leaf", "parent")

    def __init__(self, leaf: bool = True):
        self.entries: List[Entry] = []
        self.leaf: bool = leaf
        self.parent: Optional[Node] = None

    def is_leaf(self) -> bool:
        return self.leaf

    def bounding_rect(self) -> Rect:
        if not self.entries:
            raise ValueError("Empty node has no bounding rectangle")
        rect = self.entries[0].rect
        for e in self.entries[1:]:
            rect = rect_union(rect, e.rect)
        return rect


class RTree:
    def __init__(self, max_entries: int = 8, min_entries: Optional[int] = None):
        if max_entries < 4:
            raise ValueError("max_entries must be >= 4")
        self.max_entries = max_entries
        self.min_entries = min_entries if min_entries is not None else max_entries // 2
        if self.min_entries < 2:
            raise ValueError("min_entries must be >= 2")
        self.root = Node(leaf=True)

    # ---------- Public API ----------
    def insert(self, rect: Rect, obj: Any) -> None:
        leaf = self._choose_leaf(self.root, rect)
        leaf.entries.append(Entry(rect, None, obj))
        self._adjust_tree(leaf)

    def search(self, rect: Rect) -> List[Any]:
        result: List[Any] = []
        self._search(self.root, rect, result)
        return result

    def __len__(self) -> int:
        return self._count_entries(self.root)

    # ---------- Internal helpers ----------
    def _count_entries(self, node: Node) -> int:
        if node.is_leaf():
            return len(node.entries)
        return sum(self._count_entries(e.child) for e in node.entries)  # type: ignore

    def _choose_leaf(self, node: Node, rect: Rect) -> Node:
        if node.is_leaf():
            return node
        # Choose entry requiring least enlargement; tie-breaker by smallest area
        best = None
        best_enlargement = float('inf')
        best_area = float('inf')
        for e in node.entries:
            enlargement = rect_enlargement(e.rect, rect)
            area = rect_area(e.rect)
            if enlargement < best_enlargement or (enlargement == best_enlargement and area < best_area):
                best = e
                best_enlargement = enlargement
                best_area = area
        assert best is not None and best.child is not None
        return self._choose_leaf(best.child, rect)

    def _adjust_tree(self, node: Node) -> None:
        while True:
            if len(node.entries) > self.max_entries:
                node1, node2 = self._split_node(node)
                if node is self.root:
                    new_root = Node(leaf=False)
                    new_root.entries.append(Entry(node1.bounding_rect(), node1))
                    new_root.entries.append(Entry(node2.bounding_rect(), node2))
                    node1.parent = new_root
                    node2.parent = new_root
                    self.root = new_root
                    break
                else:
                    parent = node.parent  # type: ignore
                    # replace old entry with node1, add node2
                    for i, e in enumerate(parent.entries):
                        if e.child is node:
                            parent.entries[i] = Entry(node1.bounding_rect(), node1)
                            break
                    parent.entries.append(Entry(node2.bounding_rect(), node2))
                    node1.parent = parent
                    node2.parent = parent
                    node = parent
            else:
                if node is self.root:
                    break
                # tighten covering rectangles on the path upward
                parent = node.parent  # type: ignore
                for e in parent.entries:
                    if e.child is node:
                        e.rect = node.bounding_rect()
                        break
                node = parent

    def _split_node(self, node: Node) -> Tuple[Node, Node]:
        # Linear split (R*-Tree style seed selection)
        entries = node.entries
        dim = len(entries[0].rect[0])
        # Find seeds
        highest_low = [float('-inf')] * dim
        lowest_high = [float('inf')] * dim
        low_idx = high_idx = -1
        for i, e in enumerate(entries):
            for d in range(dim):
                lo, hi = e.rect[0][d], e.rect[1][d]
                if lo > highest_low[d]:
                    highest_low[d] = lo
                    low_idx = i
                if hi < lowest_high[d]:
                    lowest_high[d] = hi
                    high_idx = i
        seed1 = entries[low_idx]
        seed2 = entries[high_idx] if high_idx != low_idx else entries[(low_idx + 1) % len(entries)]

        group1 = Node(leaf=node.leaf)
        group2 = Node(leaf=node.leaf)
        group1.entries.append(seed1)
        group2.entries.append(seed2)

        # Remove seeds from list
        remaining = [e for e in entries if e not in (seed1, seed2)]

        while remaining:
            # Check if one group can take all remaining entries to satisfy min_entries
            if (len(group1.entries) + len(remaining)) == self.min_entries:
                group1.entries.extend(remaining)
                break
            if (len(group2.entries) + len(remaining)) == self.min_entries:
                group2.entries.extend(remaining)
                break

            # Pick next entry
            best_entry = None
            best_diff = float('-inf')
            best_group = None
            for e in remaining:
                d1 = rect_enlargement(group1.bounding_rect(), e.rect)
                d2 = rect_enlargement(group2.bounding_rect(), e.rect)
                diff = abs(d1 - d2)
                if diff > best_diff:
                    best_diff = diff
                    best_entry = e
                    best_group = 1 if d1 < d2 else 2
            assert best_entry is not None
            if best_group == 1:
                group1.entries.append(best_entry)
            else:
                group2.entries.append(best_entry)
            remaining.remove(best_entry)

        # Set parent links later by caller
        return group1, group2

    def _search(self, node: Node, rect: Rect, result: List[Any]) -> None:
        for e in node.entries:
            if rect_intersect(e.rect, rect):
                if node.is_leaf():
                    result.append(e.obj)
                else:
                    self._search(e.child, rect, result)  # type: ignore
