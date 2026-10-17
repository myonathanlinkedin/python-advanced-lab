from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional

BBox = Tuple[Tuple[float, ...], Tuple[float, ...]]  # (min_coords, max_coords)


def _bbox_area(bbox: BBox) -> float:
    mins, maxs = bbox
    area = 1.0
    for lo, hi in zip(mins, maxs):
        area *= max(0.0, hi - lo)
    return area


def _bbox_union(b1: BBox, b2: BBox) -> BBox:
    mins1, maxs1 = b1
    mins2, maxs2 = b2
    mins = tuple(min(a, b) for a, b in zip(mins1, mins2))
    maxs = tuple(max(a, b) for a, b in zip(maxs1, maxs2))
    return mins, maxs


def _bbox_enlargement(bbox: BBox, added: BBox) -> float:
    return _bbox_area(_bbox_union(bbox, added)) - _bbox_area(bbox)


def _bbox_intersect(b1: BBox, b2: BBox) -> bool:
    mins1, maxs1 = b1
    mins2, maxs2 = b2
    return all(lo1 <= hi2 and lo2 <= hi1 for lo1, hi1, lo2, hi2 in zip(mins1, maxs1, mins2, maxs2))


@dataclass
class _Node:
    is_leaf: bool
    entries: List[Tuple[BBox, Any]] = field(default_factory=list)  # (bbox, child) where child is data or _Node
    parent: Optional[_Node] = None

    def bbox(self) -> BBox:
        if not self.entries:
            raise ValueError("Empty node has no bounding box")
        cur = self.entries[0][0]
        for b, _ in self.entries[1:]:
            cur = _bbox_union(cur, b)
        return cur


class RTree:
    """
    Simple R‑Tree implementation using the linear split algorithm.
    Supports insertion and range search for axis‑aligned bounding boxes.
    """

    def __init__(self, max_entries: int = 8, min_entries: Optional[int] = None) -> None:
        if max_entries < 4:
            raise ValueError("max_entries must be >= 4")
        self.max_entries = max_entries
        self.min_entries = min_entries if min_entries is not None else max_entries // 2
        if self.min_entries < 2:
            raise ValueError("min_entries must be >= 2")
        self.root = _Node(is_leaf=True)

    # ---------- Public API ----------
    def insert(self, bbox: BBox, obj: Any) -> None:
        leaf = self._choose_leaf(self.root, bbox)
        leaf.entries.append((bbox, obj))
        self._adjust_tree(leaf)

    def search(self, query: BBox) -> List[Any]:
        """Return all objects whose stored bbox intersects *query*."""
        result: List[Any] = []
        self._search_node(self.root, query, result)
        return result

    # ---------- Internal helpers ----------
    def _choose_leaf(self, node: _Node, bbox: BBox) -> _Node:
        if node.is_leaf:
            return node
        # Choose entry requiring least enlargement; tie‑break by smallest area
        best_child = None
        best_enlargement = float('inf')
        best_area = float('inf')
        for child_bbox, child in node.entries:
            enlargement = _bbox_enlargement(child_bbox, bbox)
            area = _bbox_area(child_bbox)
            if enlargement < best_enlargement or (enlargement == best_enlargement and area < best_area):
                best_enlargement = enlargement
                best_area = area
                best_child = child
        assert isinstance(best_child, _Node)  # for mypy
        return self._choose_leaf(best_child, bbox)

    def _adjust_tree(self, node: _Node) -> None:
        while True:
            if len(node.entries) > self.max_entries:
                node1, node2 = self._split_node(node)
                if node.parent is None:
                    # Grow tree height
                    new_root = _Node(is_leaf=False, entries=[
                        (node1.bbox(), node1),
                        (node2.bbox(), node2)
                    ])
                    node1.parent = new_root
                    node2.parent = new_root
                    self.root = new_root
                    break
                else:
                    # Replace node entry in parent with node1 and add node2
                    parent = node.parent
                    # remove old entry
                    for i, (_, child) in enumerate(parent.entries):
                        if child is node:
                            del parent.entries[i]
                            break
                    parent.entries.append((node1.bbox(), node1))
                    parent.entries.append((node2.bbox(), node2))
                    node1.parent = parent
                    node2.parent = parent
                    node = parent
                    continue
            else:
                # Propagate bbox changes upward
                if node.parent is not None:
                    for i, (b, child) in enumerate(node.parent.entries):
                        if child is node:
                            node.parent.entries[i] = (node.bbox(), node)
                            break
                break

    def _split_node(self, node: _Node) -> Tuple[_Node, _Node]:
        # Linear split: pick seeds farthest apart on any dimension
        dim = len(node.entries[0][0][0])
        highest_low = [float('-inf')] * dim
        lowest_high = [float('inf')] * dim
        low_idx = high_idx = -1

        for idx, (bbox, _) in enumerate(node.entries):
            mins, maxs = bbox
            for d in range(dim):
                if mins[d] > highest_low[d]:
                    highest_low[d] = mins[d]
                    low_idx = idx
                if maxs[d] < lowest_high[d]:
                    lowest_high[d] = maxs[d]
                    high_idx = idx

        # If the same entry selected, pick first two distinct entries
        if low_idx == high_idx:
            low_idx = 0
            high_idx = 1

        seed1 = node.entries[low_idx]
        seed2 = node.entries[high_idx]

        group1 = [_Node(is_leaf=node.is_leaf, entries=[seed1])]
        group2 = [_Node(is_leaf=node.is_leaf, entries=[seed2])]

        # Remove seeds from consideration
        remaining = [e for i, e in enumerate(node.entries) if i not in (low_idx, high_idx)]

        # Helper to compute group bbox
        def group_bbox(g: List[_Node]) -> BBox:
            b = g[0].entries[0][0]
            for n in g:
                for entry_bbox, _ in n.entries:
                    b = _bbox_union(b, entry_bbox)
            return b

        while remaining:
            if (len(group1[0].entries) + len(remaining)) == self.min_entries:
                for e in remaining:
                    group1[0].entries.append(e)
                break
            if (len(group2[0].entries) + len(remaining)) == self.min_entries:
                for e in remaining:
                    group2[0].entries.append(e)
                break

            entry = remaining.pop(0)
            b1 = _bbox_enlargement(group_bbox(group1), entry[0])
            b2 = _bbox_enlargement(group_bbox(group2), entry[0])
            if b1 < b2:
                group1[0].entries.append(entry)
            elif b2 < b1:
                group2[0].entries.append(entry)
            else:
                # Tie‑break by smaller area
                area1 = _bbox_area(group_bbox(group1))
                area2 = _bbox_area(group_bbox(group2))
                if area1 < area2:
                    group1[0].entries.append(entry)
                else:
                    group2[0].entries.append(entry)

        node1 = group1[0]
        node2 = group2[0]
        node1.parent = node.parent
        node2.parent = node.parent
        return node1, node2

    def _search_node(self, node: _Node, query: BBox, out: List[Any]) -> None:
        for entry_bbox, child in node.entries:
            if not _bbox_intersect(entry_bbox, query):
                continue
            if node.is_leaf:
                out.append(child)
            else:
                self._search_node(child, query, out)


__all__ = ["RTree", "BBox"]
