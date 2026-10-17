from __future__ import annotations
from typing import Any, List, Tuple, Optional

Rect = Tuple[Tuple[float, ...], Tuple[float, ...]]  # (min_coords, max_coords)


def rect_area(rect: Rect) -> float:
    mins, maxs = rect
    area = 1.0
    for lo, hi in zip(mins, maxs):
        area *= max(0.0, hi - lo)
    return area


def rect_combine(a: Rect, b: Rect) -> Rect:
    mins_a, maxs_a = a
    mins_b, maxs_b = b
    mins = tuple(min(x, y) for x, y in zip(mins_a, mins_b))
    maxs = tuple(max(x, y) for x, y in zip(maxs_a, maxs_b))
    return mins, maxs


def rect_enlargement(rect: Rect, other: Rect) -> float:
    return rect_area(rect_combine(rect, other)) - rect_area(rect)


def rect_intersect(a: Rect, b: Rect) -> bool:
    mins_a, maxs_a = a
    mins_b, maxs_b = b
    for lo_a, hi_a, lo_b, hi_b in zip(mins_a, maxs_a, mins_b, maxs_b):
        if lo_a > hi_b or lo_b > hi_a:
            return False
    return True


class Node:
    __slots__ = ("leaf", "entries", "mbr")

    def __init__(self, leaf: bool):
        self.leaf: bool = leaf
        self.entries: List[Tuple[Rect, Any]] = []  # (rect, child) or (rect, obj)
        self.mbr: Optional[Rect] = None

    def update_mbr(self) -> None:
        if not self.entries:
            self.mbr = None
            return
        cur = self.entries[0][0]
        for r, _ in self.entries[1:]:
            cur = rect_combine(cur, r)
        self.mbr = cur


class RTree:
    def __init__(self, max_entries: int = 8, min_entries: Optional[int] = None):
        if max_entries < 4:
            raise ValueError("max_entries must be >= 4")
        self.max_entries: int = max_entries
        self.min_entries: int = min_entries if min_entries is not None else max_entries // 2
        if self.min_entries < 2:
            self.min_entries = 2
        self.root: Node = Node(leaf=True)

    # ---------- Public API ----------
    def insert(self, rect: Rect, obj: Any) -> None:
        path: List[Node] = []
        node = self.root
        while not node.leaf:
            path.append(node)
            node = self._choose_subtree(node, rect)
        node.entries.append((rect, obj))
        node.update_mbr()
        if len(node.entries) > self.max_entries:
            new_node = self._split_node(node)
            self._handle_split(path, node, new_node)
        else:
            self._propagate_mbr(path, node)

    def search(self, rect: Rect) -> List[Any]:
        result: List[Any] = []

        def recurse(n: Node) -> None:
            for entry_rect, child in n.entries:
                if rect_intersect(entry_rect, rect):
                    if n.leaf:
                        result.append(child)
                    else:
                        recurse(child)

        recurse(self.root)
        return result

    # ---------- Internal helpers ----------
    def _choose_subtree(self, node: Node, rect: Rect) -> Node:
        best = None
        best_enlargement = None
        best_area = None
        for entry_rect, child in node.entries:
            enlargement = rect_enlargement(entry_rect, rect)
            area = rect_area(entry_rect)
            if (best_enlargement is None or enlargement < best_enlargement or
                (enlargement == best_enlargement and (area < best_area))):
                best = child
                best_enlargement = enlargement
                best_area = area
        return best  # type: ignore

    def _split_node(self, node: Node) -> Node:
        # Quadratic split
        entries = node.entries
        node.entries = []

        # 1. Pick seeds
        max_d = -1.0
        seed1 = seed2 = None
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                r1, _ = entries[i]
                r2, _ = entries[j]
                combined = rect_combine(r1, r2)
                d = rect_area(combined) - rect_area(r1) - rect_area(r2)
                if d > max_d:
                    max_d = d
                    seed1, seed2 = i, j
        assert seed1 is not None and seed2 is not None
        group1 = [entries[seed1]]
        group2 = [entries[seed2]]

        # Remove seeds
        remaining = [e for k, e in enumerate(entries) if k not in (seed1, seed2)]

        def mbr(group: List[Tuple[Rect, Any]]) -> Rect:
            cur = group[0][0]
            for r, _ in group[1:]:
                cur = rect_combine(cur, r)
            return cur

        while remaining:
            if (len(group1) + len(remaining)) == self.min_entries:
                group1.extend(remaining)
                break
            if (len(group2) + len(remaining)) == self.min_entries:
                group2.extend(remaining)
                break

            # Choose next entry
            diff = -1.0
            chosen = None
            target_group = None
            for e in remaining:
                r, _ = e
                d1 = rect_enlargement(mbr(group1), r)
                d2 = rect_enlargement(mbr(group2), r)
                d = abs(d1 - d2)
                if d > diff:
                    diff = d
                    chosen = e
                    target_group = 1 if d1 < d2 else 2
                elif d == diff:
                    # tie‑break by smaller area
                    area1 = rect_area(mbr(group1))
                    area2 = rect_area(mbr(group2))
                    if area1 < area2:
                        target_group = 1
                    else:
                        target_group = 2
            remaining.remove(chosen)  # type: ignore
            if target_group == 1:
                group1.append(chosen)  # type: ignore
            else:
                group2.append(chosen)  # type: ignore

        node.entries = group1
        node.update_mbr()
        new_node = Node(node.leaf)
        new_node.entries = group2
        new_node.update_mbr()
        return new_node

    def _handle_split(self, path: List[Node], node: Node, new_node: Node) -> None:
        while True:
            if not path:
                # Root was split
                new_root = Node(leaf=False)
                new_root.entries = [(node.mbr, node), (new_node.mbr, new_node)]
                new_root.update_mbr()
                self.root = new_root
                break
            parent = path.pop()
            # Replace old entry for node
            for i, (r, child) in enumerate(parent.entries):
                if child is node:
                    parent.entries[i] = (node.mbr, node)
                    break
            # Add entry for new_node
            parent.entries.append((new_node.mbr, new_node))
            parent.update_mbr()
            if len(parent.entries) > self.max_entries:
                node = parent
                new_node = self._split_node(parent)
                continue
            else:
                self._propagate_mbr(path, parent)
                break

    def _propagate_mbr(self, path: List[Node], node: Node) -> None:
        while path:
            parent = path.pop()
            for i, (r, child) in enumerate(parent.entries):
                if child is node:
                    parent.entries[i] = (node.mbr, node)
                    break
            parent.update_mbr()
            node = parent

    def _propagate_mbr(self, path: List[Node], node: Node) -> None:
        self._propagate_mbr(path, node)  # placeholder to avoid unused warning
