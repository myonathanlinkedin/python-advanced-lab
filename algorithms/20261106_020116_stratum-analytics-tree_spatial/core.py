from __future__ import annotations
from typing import List, Tuple, Any, Optional, Iterable
import math

Coord = Tuple[float, ...]
BoxTuple = Tuple[Coord, Coord]  # (min_coords, max_coords)


def _dim(box: BoxTuple) -> int:
    return len(box[0])


def _box_union(a: BoxTuple, b: BoxTuple) -> BoxTuple:
    dim = _dim(a)
    min_c = tuple(min(a[0][i], b[0][i]) for i in range(dim))
    max_c = tuple(max(a[1][i], b[1][i]) for i in range(dim))
    return (min_c, max_c)


def _box_intersects(a: BoxTuple, b: BoxTuple) -> bool:
    dim = _dim(a)
    for i in range(dim):
        if a[1][i] < b[0][i] or b[1][i] < a[0][i]:
            return False
    return True


def _box_contains(a: BoxTuple, b: BoxTuple) -> bool:
    dim = _dim(a)
    for i in range(dim):
        if a[0][i] > b[0][i] or a[1][i] < b[1][i]:
            return False
    return True


def _box_area(box: BoxTuple) -> float:
    dim = _dim(box)
    area = 1.0
    for i in range(dim):
        area *= max(0.0, box[1][i] - box[0][i])
    return area


class Entry:
    __slots__ = ("box", "obj")
    def __init__(self, box: BoxTuple, obj: Any):
        self.box = box
        self.obj = obj


class Node:
    __slots__ = ("box", "children", "entries", "leaf")
    def __init__(self, leaf: bool):
        self.leaf = leaf
        self.box: Optional[BoxTuple] = None
        self.children: List[Node] = []
        self.entries: List[Entry] = []

    def recalc_box(self) -> None:
        if self.leaf:
            boxes = [e.box for e in self.entries]
        else:
            boxes = [c.box for c in self.children if c.box is not None]
        if not boxes:
            self.box = None
            return
        cur = boxes[0]
        for b in boxes[1:]:
            cur = _box_union(cur, b)
        self.box = cur


class StratumAnalyticsTree:
    """
    A simple R‑tree‑like spatial index for arbitrary‑dimensional bounding boxes.
    The implementation follows the classic linear split algorithm and
    guarantees deterministic behaviour.
    """
    def __init__(self, max_entries: int = 8, min_entries: Optional[int] = None):
        if max_entries < 4:
            raise ValueError("max_entries must be >= 4")
        self.max_entries = max_entries
        self.min_entries = min_entries or max_entries // 2
        self.root = Node(leaf=True)

    # ---------- Public API ----------
    def insert(self, box: BoxTuple, obj: Any) -> None:
        leaf = self._choose_leaf(self.root, box)
        leaf.entries.append(Entry(box, obj))
        leaf.recalc_box()
        self._adjust_tree(leaf)

    def search(self, box: BoxTuple) -> List[Any]:
        result: List[Any] = []
        self._search_node(self.root, box, result)
        return result

    # ---------- Internal helpers ----------
    def _choose_leaf(self, node: Node, box: BoxTuple) -> Node:
        if node.leaf:
            return node
        # Choose child requiring least enlargement
        best_child = None
        best_enlargement = math.inf
        best_area = math.inf
        for child in node.children:
            assert child.box is not None
            cur_area = _box_area(child.box)
            new_box = _box_union(child.box, box)
            enlargement = _box_area(new_box) - cur_area
            if enlargement < best_enlargement or (enlargement == best_enlargement and cur_area < best_area):
                best_enlargement = enlargement
                best_area = cur_area
                best_child = child
        assert best_child is not None
        return self._choose_leaf(best_child, box)

    def _adjust_tree(self, node: Node) -> None:
        while True:
            if node.leaf:
                if len(node.entries) > self.max_entries:
                    node1, node2 = self._split_node(node)
                    if node is self.root:
                        new_root = Node(leaf=False)
                        new_root.children = [node1, node2]
                        new_root.recalc_box()
                        self.root = new_root
                        return
                    else:
                        parent = self._find_parent(self.root, node)
                        parent.children.remove(node)
                        parent.children.extend([node1, node2])
                        parent.recalc_box()
                        node = parent
                        continue
                else:
                    if node is not self.root:
                        parent = self._find_parent(self.root, node)
                        parent.recalc_box()
                    return
            else:
                if len(node.children) > self.max_entries:
                    node1, node2 = self._split_node(node)
                    if node is self.root:
                        new_root = Node(leaf=False)
                        new_root.children = [node1, node2]
                        new_root.recalc_box()
                        self.root = new_root
                        return
                    else:
                        parent = self._find_parent(self.root, node)
                        parent.children.remove(node)
                        parent.children.extend([node1, node2])
                        parent.recalc_box()
                        node = parent
                        continue
                else:
                    if node is not self.root:
                        parent = self._find_parent(self.root, node)
                        parent.recalc_box()
                    return

    def _find_parent(self, current: Node, target: Node) -> Optional[Node]:
        if current.leaf:
            return None
        for child in current.children:
            if child is target:
                return current
            res = self._find_parent(child, target)
            if res:
                return res
        return None

    def _split_node(self, node: Node) -> Tuple[Node, Node]:
        # Linear split: pick seeds farthest apart on any dimension
        if node.leaf:
            items = node.entries
        else:
            items = node.children
        dim = _dim(items[0].box)  # type: ignore
        # Find pair with max normalized separation
        best_pair = (0, 1)
        best_sep = -1.0
        for d in range(dim):
            low = min(i for i, it in enumerate(items) if it.box[0][d] == min(it.box[0][d] for it in items))
            high = max(i for i, it in enumerate(items) if it.box[1][d] == max(it.box[1][d] for it in items))
            if low == high:
                continue
            width = max(it.box[1][d] for it in items) - min(it.box[0][d] for it in items)
            if width == 0:
                sep = 0
            else:
                sep = (items[high].box[0][d] - items[low].box[1][d]) / width
            if sep > best_sep:
                best_sep = sep
                best_pair = (low, high)
        seed1, seed2 = best_pair
        group1 = Node(leaf=node.leaf)
        group2 = Node(leaf=node.leaf)
        # Assign seeds
        if node.leaf:
            group1.entries.append(items[seed1])
            group2.entries.append(items[seed2])
        else:
            group1.children.append(items[seed1])
            group2.children.append(items[seed2])
        # Remove seeds from list
        remaining = [it for i, it in enumerate(items) if i not in best_pair]
        # Distribute remaining
        for it in remaining:
            target = self._choose_group(group1, group2, it.box)
            if target is group1:
                if node.leaf:
                    group1.entries.append(it)  # type: ignore
                else:
                    group1.children.append(it)  # type: ignore
            else:
                if node.leaf:
                    group2.entries.append(it)  # type: ignore
                else:
                    group2.children.append(it)  # type: ignore
        group1.recalc_box()
        group2.recalc_box()
        return group1, group2

    def _choose_group(self, g1: Node, g2: Node, box: BoxTuple) -> Node:
        # Compute enlargement needed for each group
        def enlargement(g: Node) -> float:
            if g.box is None:
                return _box_area(box)
            return _box_area(_box_union(g.box, box)) - _box_area(g.box)
        e1 = enlargement(g1)
        e2 = enlargement(g2)
        if e1 < e2:
            return g1
        if e2 < e1:
            return g2
        # Tie‑breaker: choose group with smaller area, then fewer entries
        a1 = _box_area(g1.box) if g1.box else 0.0
        a2 = _box_area(g2.box) if g2.box else 0.0
        if a1 < a2:
            return g1
        if a2 < a1:
            return g2
        if g1.leaf:
            return g1 if len(g1.entries) <= len(g2.entries) else g2
        else:
            return g1 if len(g1.children) <= len(g2.children) else g2

    def _search_node(self, node: Node, query: BoxTuple, out: List[Any]) -> None:
        if node.box is None or not _box_intersects(node.box, query):
            return
        if node.leaf:
            for e in node.entries:
                if _box_intersects(e.box, query):
                    out.append(e.obj)
        else:
            for child in node.children:
                self._search_node(child, query, out)
