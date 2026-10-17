from __future__ import annotations
from typing import List, Tuple, Any, Optional, Callable

Rect = Tuple[Tuple[float, ...], Tuple[float, ...]]  # (min_coords, max_coords)


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


def rect_enlargement(base: Rect, added: Rect) -> float:
    return rect_area(rect_union(base, added)) - rect_area(base)


def rect_intersects(a: Rect, b: Rect) -> bool:
    min_a, max_a = a
    min_b, max_b = b
    return all(min_a[i] <= max_b[i] and max_a[i] >= min_b[i] for i in range(len(min_a)))


class Node:
    def __init__(self, leaf: bool = True):
        self.leaf: bool = leaf
        self.entries: List[Tuple[Rect, Any]] = []  # (rect, obj) where obj is child Node if not leaf
        self.parent: Optional[Node] = None

    def is_overflow(self, max_entries: int) -> bool:
        return len(self.entries) > max_entries

    def bounding_rect(self) -> Rect:
        if not self.entries:
            raise ValueError("Empty node has no bounding rectangle")
        rect = self.entries[0][0]
        for e_rect, _ in self.entries[1:]:
            rect = rect_union(rect, e_rect)
        return rect

    def __repr__(self) -> str:
        typ = "Leaf" if self.leaf else "Internal"
        return f"<{typ}Node entries={len(self.entries)}>"


class RTree:
    def __init__(self, max_entries: int = 8, min_entries: Optional[int] = None):
        if max_entries < 4:
            raise ValueError("max_entries must be >= 4")
        self.max_entries = max_entries
        self.min_entries = min_entries or max_entries // 2
        if self.min_entries < 2:
            self.min_entries = 2
        self.root = Node(leaf=True)

    # ---------- Public API ----------
    def insert(self, rect: Rect, obj: Any) -> None:
        leaf = self._choose_leaf(self.root, rect)
        leaf.entries.append((rect, obj))
        self._adjust_tree(leaf)

    def search(self, rect: Rect) -> List[Any]:
        result: List[Any] = []
        self._search(self.root, rect, result)
        return result

    # ---------- Internal helpers ----------
    def _choose_leaf(self, node: Node, rect: Rect) -> Node:
        if node.leaf:
            return node
        # Choose the entry whose rectangle needs least enlargement to include rect
        best_child = None
        min_enlargement = float('inf')
        min_area = float('inf')
        for entry_rect, child in node.entries:
            enlargement = rect_enlargement(entry_rect, rect)
            area = rect_area(entry_rect)
            if (enlargement < min_enlargement or
                (enlargement == min_enlargement and area < min_area)):
                min_enlargement = enlargement
                min_area = area
                best_child = child
        assert isinstance(best_child, Node)
        return self._choose_leaf(best_child, rect)

    def _adjust_tree(self, node: Node) -> None:
        while True:
            if node.is_overflow(self.max_entries):
                node1, node2 = self._split_node(node)
                if node is self.root:
                    new_root = Node(leaf=False)
                    new_root.entries.append((node1.bounding_rect(), node1))
                    new_root.entries.append((node2.bounding_rect(), node2))
                    node1.parent = new_root
                    node2.parent = new_root
                    self.root = new_root
                    break
                else:
                    parent = node.parent
                    assert parent is not None
                    # replace old entry with node1, add node2
                    for i, (r, child) in enumerate(parent.entries):
                        if child is node:
                            parent.entries[i] = (node1.bounding_rect(), node1)
                            break
                    parent.entries.append((node2.bounding_rect(), node2))
                    node1.parent = parent
                    node2.parent = parent
                    node = parent
                    continue
            else:
                if node is not self.root:
                    parent = node.parent
                    assert parent is not None
                    # tighten bounding rectangle in parent entry
                    for i, (r, child) in enumerate(parent.entries):
                        if child is node:
                            parent.entries[i] = (node.bounding_rect(), node)
                            break
                if node is self.root:
                    break
                node = node.parent

    def _split_node(self, node: Node) -> Tuple[Node, Node]:
        # Linear split: pick seeds with greatest dead space
        entries = node.entries
        dim = len(entries[0][0][0])
        # Find pair with max normalized separation
        highest_sep = -1.0
        seed1_idx = seed2_idx = -1
        for d in range(dim):
            low_vals = [e[0][0][d] for e in entries]
            high_vals = [e[0][1][d] for e in entries]
            min_low = min(low_vals)
            max_high = max(high_vals)
            width = max_high - min_low
            if width == 0:
                continue
            # find entry with lowest low and highest high
            low_idx = low_vals.index(min_low)
            high_idx = high_vals.index(max_high)
            sep = (high_vals[high_idx] - low_vals[low_idx]) / width
            if sep > highest_sep:
                highest_sep = sep
                seed1_idx, seed2_idx = low_idx, high_idx
        if seed1_idx == seed2_idx:
            # fallback: first two entries
            seed1_idx, seed2_idx = 0, 1
        group1 = Node(leaf=node.leaf)
        group2 = Node(leaf=node.leaf)
        seeds = [entries[seed1_idx], entries[seed2_idx]]
        group1.entries.append(seeds[0])
        group2.entries.append(seeds[1])
        # Remove seeds from list
        remaining = [e for i, e in enumerate(entries) if i not in (seed1_idx, seed2_idx)]

        while remaining:
            if (len(group1.entries) + len(remaining)) == self.min_entries:
                group1.entries.extend(remaining)
                break
            if (len(group2.entries) + len(remaining)) == self.min_entries:
                group2.entries.extend(remaining)
                break
            # pick next entry
            entry = remaining.pop(0)
            rect, _ = entry
            d1 = rect_enlargement(group1.bounding_rect(), rect)
            d2 = rect_enlargement(group2.bounding_rect(), rect)
            if d1 < d2:
                group1.entries.append(entry)
            elif d2 < d1:
                group2.entries.append(entry)
            else:
                # tie-breaker: choose group with smaller area
                if rect_area(group1.bounding_rect()) < rect_area(group2.bounding_rect()):
                    group1.entries.append(entry)
                else:
                    group2.entries.append(entry)

        # Set parent links later by caller
        return group1, group2

    def _search(self, node: Node, rect: Rect, result: List[Any]) -> None:
        for entry_rect, child in node.entries:
            if not rect_intersects(entry_rect, rect):
                continue
            if node.leaf:
                result.append(child)
            else:
                assert isinstance(child, Node)
                self._search(child, rect, result)
