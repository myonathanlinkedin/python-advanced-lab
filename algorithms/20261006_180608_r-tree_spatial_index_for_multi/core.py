from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional

@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned bounding rectangle in N dimensions."""
    mins: Tuple[float, ...]
    maxs: Tuple[float, ...]

    def __post_init__(self):
        if len(self.mins) != len(self.maxs):
            raise ValueError("mins and maxs must have same dimensionality")
        for mi, ma in zip(self.mins, self.maxs):
            if mi > ma:
                raise ValueError("Each min must be <= corresponding max")

    def area(self) -> float:
        prod = 1.0
        for mi, ma in zip(self.mins, self.maxs):
            prod *= ma - mi
        return prod

    def enlarge_to_include(self, other: Rectangle) -> Rectangle:
        new_mins = tuple(min(a, b) for a, b in zip(self.mins, other.mins))
        new_maxs = tuple(max(a, b) for a, b in zip(self.maxs, other.maxs))
        return Rectangle(new_mins, new_maxs)

    def intersects(self, other: Rectangle) -> bool:
        for mi, ma, omi, oma in zip(self.mins, self.maxs, other.mins, other.maxs):
            if ma < omi or oma < mi:
                return False
        return True

    def contains(self, other: Rectangle) -> bool:
        return all(mi <= omi and ma >= oma for mi, ma, omi, oma in zip(
            self.mins, self.maxs, other.mins, other.maxs))

@dataclass
class Entry:
    rect: Rectangle
    child: Optional['Node'] = None
    data: Any = None

@dataclass
class Node:
    is_leaf: bool
    entries: List[Entry] = field(default_factory=list)

    def mbr(self) -> Rectangle:
        if not self.entries:
            raise ValueError("Node has no entries")
        mbr = self.entries[0].rect
        for e in self.entries[1:]:
            mbr = mbr.enlarge_to_include(e.rect)
        return mbr

class RTree:
    MAX_ENTRIES: int = 4
    MIN_ENTRIES: int = 2

    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self.root = Node(is_leaf=True)

    def insert(self, rect: Rectangle, data: Any) -> None:
        leaf = self._choose_leaf(self.root, rect)
        leaf.entries.append(Entry(rect, data=data))
        if len(leaf.entries) > self.MAX_ENTRIES:
            self._split_node(leaf)

    def _choose_leaf(self, node: Node, rect: Rectangle) -> Node:
        if node.is_leaf:
            return node
        # Choose child whose MBR needs least enlargement
        best = None
        best_enlargement = None
        for e in node.entries:
            current_area = e.rect.area()
            enlarged = e.rect.enlarge_to_include(rect)
            enlargement = enlarged.area() - current_area
            if best is None or enlargement < best_enlargement or (
                enlargement == best_enlargement and current_area < best.rect.area()
            ):
                best = e
                best_enlargement = enlargement
        return self._choose_leaf(best.child, rect)

    def _split_node(self, node: Node) -> None:
        group1, group2 = self._quadratic_split(node.entries)
        node.entries = group1
        if node is self.root:
            new_root = Node(is_leaf=False)
            new_root.entries.append(Entry(node.mbr(), child=node))
            new_root.entries.append(Entry(self._mbr_of_entries(group2), child=Node(is_leaf=node.is_leaf, entries=group2)))
            self.root = new_root
        else:
            parent = self._find_parent(self.root, node)
            parent.entries = [e for e in parent.entries if e.child is not node]
            parent.entries.append(Entry(node.mbr(), child=node))
            parent.entries.append(Entry(self._mbr_of_entries(group2), child=Node(is_leaf=node.is_leaf, entries=group2)))
            if len(parent.entries) > self.MAX_ENTRIES:
                self._split_node(parent)

    def _quadratic_split(self, entries: List[Entry]) -> Tuple[List[Entry], List[Entry]]:
        def pick_seeds(es: List[Entry]) -> Tuple[Entry, Entry]:
            max_d = -1.0
            seed1 = seed2 = None
            for i in range(len(es)):
                for j in range(i + 1, len(es)):
                    rect1, rect2 = es[i].rect, es[j].rect
                    d = rect1.enlarge_to_include(rect2).area() - rect1.area() - rect2.area()
                    if d > max_d:
                        max_d = d
                        seed1, seed2 = es[i], es[j]
            return seed1, seed2

        seed1, seed2 = pick_seeds(entries)
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
            e = remaining.pop()
            mbr1 = self._mbr_of_entries(group1).enlarge_to_include(e.rect)
            mbr2 = self._mbr_of_entries(group2).enlarge_to_include(e.rect)
            diff1 = mbr1.area() - self._mbr_of_entries(group1).area()
            diff2 = mbr2.area() - self._mbr_of_entries(group2).area()
            if diff1 < diff2:
                group1.append(e)
            elif diff2 < diff1:
                group2.append(e)
            else:
                if self._mbr_of_entries(group1).area() < self._mbr_of_entries(group2).area():
                    group1.append(e)
                else:
                    group2.append(e)
        return group1, group2

    def _mbr_of_entries(self, entries: List[Entry]) -> Rectangle:
        mbr = entries[0].rect
        for e in entries[1:]:
            mbr = mbr.enlarge_to_include(e.rect)
        return mbr

    def _find_parent(self, current: Node, child: Node) -> Node:
        if current.is_leaf:
            raise ValueError("Root has no parent")
        for e in current.entries:
            if e.child is child:
                return current
            if not e.child.is_leaf:
                try:
                    return self._find_parent(e.child, child)
                except ValueError:
                    continue
        raise ValueError("Parent not found")

    def search(self, rect: Rectangle) -> List[Any]:
        return self._search(self.root, rect)

    def _search(self, node: Node, rect: Rectangle) -> List[Any]:
        results = []
        for e in node.entries:
            if e.rect.intersects(rect):
                if node.is_leaf:
                    results.append(e.data)
                else:
                    results.extend(self._search(e.child, rect))
        return results
