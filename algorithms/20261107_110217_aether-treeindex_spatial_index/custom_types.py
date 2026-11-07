from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Sequence, Optional


@dataclass(frozen=True)
class BoundingBox:
    """
    Axis-aligned bounding box in an N‑dimensional space.
    ``mins`` and ``maxs`` must have the same length and ``mins[i] <= maxs[i]`` for all i.
    """
    mins: Tuple[float, ...]
    maxs: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.mins) != len(self.maxs):
            raise ValueError("mins and maxs must have the same dimensionality")
        for i, (mi, ma) in enumerate(zip(self.mins, self.maxs)):
            if mi > ma:
                raise ValueError(f"mins[{i}] > maxs[{i}] ({mi} > {ma})")

    @property
    def dim(self) -> int:
        return len(self.mins)

    def intersects(self, other: BoundingBox) -> bool:
        """Return True iff this box overlaps (including touching) ``other``."""
        for a_min, a_max, b_min, b_max in zip(self.mins, self.maxs, other.mins, other.maxs):
            if a_max < b_min or b_max < a_min:
                return False
        return True

    def contains_point(self, point: Sequence[float]) -> bool:
        """Return True iff ``point`` lies inside (or on the boundary of) the box."""
        if len(point) != self.dim:
            raise ValueError("Point dimensionality mismatch")
        return all(mn <= p <= mx for p, mn, mx in zip(point, self.mins, self.maxs))

    def enlarge_to_include(self, other: BoundingBox) -> BoundingBox:
        """Return a new BoundingBox that is the minimal bounding box containing both."""
        new_mins = tuple(min(a, b) for a, b in zip(self.mins, other.mins))
        new_maxs = tuple(max(a, b) for a, b in zip(self.maxs, other.maxs))
        return BoundingBox(new_mins, new_maxs)

    def area(self) -> float:
        """Hyper‑volume of the box (product of edge lengths)."""
        prod = 1.0
        for mn, mx in zip(self.mins, self.maxs):
            prod *= max(0.0, mx - mn)
        return prod

    def margin(self) -> float:
        """Sum of edge lengths; useful for split heuristics."""
        return sum(max(0.0, mx - mn) for mn, mx in zip(self.mins, self.maxs))

    def __repr__(self) -> str:
        return f"BoundingBox(mins={self.mins}, maxs={self.maxs})"


@dataclass
class LeafEntry:
    """A payload stored in a leaf node."""
    bbox: BoundingBox
    obj: Any


@dataclass
class Node:
    """Tree node – either leaf (holds LeafEntry) or internal (holds child Nodes)."""
    leaf: bool
    entries: List[LeafEntry] = field(default_factory=list)          # used when leaf == True
    children: List[Node] = field(default_factory=list)              # used when leaf == False
    bbox: Optional[BoundingBox] = None                               # bounding box covering all entries/children
    parent: Optional[Node] = None

    def is_overflow(self, max_entries: int) -> bool:
        if self.leaf:
            return len(self.entries) > max_entries
        else:
            return len(self.children) > max_entries

    def recalc_bbox(self) -> None:
        """Recompute the node's bounding box from its contents."""
        if self.leaf:
            if not self.entries:
                self.bbox = None
                return
            b = self.entries[0].bbox
            for e in self.entries[1:]:
                b = b.enlarge_to_include(e.bbox)
            self.bbox = b
        else:
            if not self.children:
                self.bbox = None
                return
            b = self.children[0].bbox
            for c in self.children[1:]:
                b = b.enlarge_to_include(c.bbox)
            self.bbox = b

    def add_entry(self, entry: LeafEntry) -> None:
        assert self.leaf, "add_entry called on internal node"
        self.entries.append(entry)
        self.bbox = entry.bbox if self.bbox is None else self.bbox.enlarge_to_include(entry.bbox)

    def add_child(self, child: Node) -> None:
        assert not self.leaf, "add_child called on leaf node"
        child.parent = self
        self.children.append(child)
        self.bbox = child.bbox if self.bbox is None else self.bbox.enlarge_to_include(child.bbox)

    def __repr__(self) -> str:
        typ = "Leaf" if self.leaf else "Internal"
        return f"<{typ}Node entries={len(self.entries)} children={len(self.children)} bbox={self.bbox}>"
