from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Any, Optional, Union


Coord = Tuple[float, ...]


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned bounding box in N dimensions."""
    min_corner: Coord
    max_corner: Coord

    def __post_init__(self) -> None:
        if len(self.min_corner) != len(self.max_corner):
            raise ValueError("Dimension mismatch between min and max corners.")
        for lo, hi in zip(self.min_corner, self.max_corner):
            if lo > hi:
                raise ValueError("Each min coordinate must be <= the corresponding max coordinate.")

    @property
    def dim(self) -> int:
        return len(self.min_corner)

    def area(self) -> float:
        """Hyper‑volume of the rectangle."""
        prod = 1.0
        for lo, hi in zip(self.min_corner, self.max_corner):
            prod *= (hi - lo)
        return prod

    def margin(self) -> float:
        """Sum of edge lengths (used by some split heuristics)."""
        return sum((hi - lo) for lo, hi in zip(self.min_corner, self.max_corner)) * 2

    def contains_point(self, point: Coord) -> bool:
        return all(lo <= p <= hi for lo, hi, p in zip(self.min_corner, self.max_corner, point))

    def intersects(self, other: Rectangle) -> bool:
        return all(not (self.max_corner[i] < other.min_corner[i] or self.min_corner[i] > other.max_corner[i])
                   for i in range(self.dim))

    def combine(self, other: Rectangle) -> Rectangle:
        """Return the minimal bounding rectangle containing both."""
        new_min = tuple(min(a, b) for a, b in zip(self.min_corner, other.min_corner))
        new_max = tuple(max(a, b) for a, b in zip(self.max_corner, other.max_corner))
        return Rectangle(new_min, new_max)

    def enlargement_needed(self, other: Rectangle) -> float:
        """Additional area required to include `other`."""
        combined = self.combine(other)
        return combined.area() - self.area()


@dataclass
class RTreeNode:
    """Node of an R‑Tree.  Leaf nodes store (Rectangle, value) pairs,
    internal nodes store (Rectangle, child_node) pairs."""
    leaf: bool
    entries: List[Tuple[Rectangle, Union[Any, "RTreeNode"]]] = field(default_factory=list)
    parent: Optional["RTreeNode"] = field(default=None, repr=False)

    def is_overflow(self, max_entries: int) -> bool:
        return len(self.entries) > max_entries

    def mbr(self) -> Rectangle:
        """Compute the minimal bounding rectangle of all entries."""
        if not self.entries:
            raise ValueError("Cannot compute MBR of an empty node.")
        rect = self.entries[0][0]
        for entry_rect, _ in self.entries[1:]:
            rect = rect.combine(entry_rect)
        return rect

    def add_entry(self, rect: Rectangle, obj: Union[Any, "RTreeNode"]) -> None:
        self.entries.append((rect, obj))
        if isinstance(obj, RTreeNode):
            obj.parent = self

    def remove_entry(self, rect: Rectangle, obj: Union[Any, "RTreeNode"]) -> None:
        self.entries.remove((rect, obj))
        if isinstance(obj, RTreeNode):
            obj.parent = None

    def __repr__(self) -> str:
        typ = "Leaf" if self.leaf else "Internal"
        return f"<RTreeNode {typ} entries={len(self.entries)}>"
