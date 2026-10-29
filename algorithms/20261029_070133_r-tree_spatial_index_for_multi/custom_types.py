from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Any, Tuple, Union, Iterable, Optional

@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in N-dimensional space."""
    mins: Tuple[float, ...]
    maxs: Tuple[float, ...]

    def __post_init__(self):
        if len(self.mins) != len(self.maxs):
            raise ValueError("mins and maxs must have the same dimensionality")
        for min_val, max_val in zip(self.mins, self.maxs):
            if min_val > max_val:
                raise ValueError("Each min must be <= corresponding max")

    def intersects(self, other: BoundingBox) -> bool:
        """Return True if this box intersects with another."""
        for min_a, max_a, min_b, max_b in zip(self.mins, self.maxs, other.mins, other.maxs):
            if max_a < min_b or max_b < min_a:
                return False
        return True

    def contains(self, other: BoundingBox) -> bool:
        """Return True if this box fully contains another."""
        for min_a, max_a, min_b, max_b in zip(self.mins, self.maxs, other.mins, other.maxs):
            if min_a > min_b or max_a < max_b:
                return False
        return True

    def union(self, other: BoundingBox) -> BoundingBox:
        """Return the minimal bounding box that contains both."""
        new_mins = tuple(min(a, b) for a, b in zip(self.mins, other.mins))
        new_maxs = tuple(max(a, b) for a, b in zip(self.maxs, other.maxs))
        return BoundingBox(new_mins, new_maxs)

    def area(self) -> float:
        """Return the hypervolume of the bounding box."""
        prod = 1.0
        for min_val, max_val in zip(self.mins, self.maxs):
            prod *= (max_val - min_val)
        return prod

@dataclass
class RTreeEntry:
    """Entry in an R-Tree node."""
    bbox: BoundingBox
    child: Optional['RTreeNode'] = None  # None for leaf entries
    data: Any = None  # Only for leaf entries

@dataclass
class RTreeNode:
    """Node in an R-Tree."""
    is_leaf: bool
    entries: List[RTreeEntry] = field(default_factory=list)

    def bbox(self) -> BoundingBox:
        """Compute the bounding box that encloses all entries."""
        if not self.entries:
            raise ValueError("Node has no entries")
        bbox = self.entries[0].bbox
        for entry in self.entries[1:]:
            bbox = bbox.union(entry.bbox)
        return bbox
