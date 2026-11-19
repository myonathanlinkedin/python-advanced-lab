from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Tuple


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """
    Axis-aligned bounding box in an N‑dimensional space.
    `mins` and `maxs` must have the same length (the dimensionality).
    """
    mins: Tuple[float, ...]
    maxs: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.mins) != len(self.maxs):
            raise ValueError("mins and maxs must have the same dimensionality")
        for lo, hi in zip(self.mins, self.maxs):
            if lo > hi:
                raise ValueError("Each min coordinate must be <= the corresponding max coordinate")

    @property
    def dim(self) -> int:
        return len(self.mins)

    def intersects(self, other: BoundingBox) -> bool:
        """Return True iff this box overlaps (including touching) `other`."""
        for a_min, a_max, b_min, b_max in zip(self.mins, self.maxs, other.mins, other.maxs):
            if a_max < b_min or b_max < a_min:
                return False
        return True

    def contains(self, other: BoundingBox) -> bool:
        """Return True iff this box completely contains `other`."""
        for a_min, a_max, b_min, b_max in zip(self.mins, self.maxs, other.mins, other.maxs):
            if a_min > b_min or a_max < b_max:
                return False
        return True

    def enlarged(self, other: BoundingBox) -> BoundingBox:
        """Return a new BoundingBox that minimally encloses both boxes."""
        new_mins = tuple(min(a, b) for a, b in zip(self.mins, other.mins))
        new_maxs = tuple(max(a, b) for a, b in zip(self.maxs, other.maxs))
        return BoundingBox(new_mins, new_maxs)

    def area(self) -> float:
        """Return the hyper‑volume (product of side lengths)."""
        prod = 1.0
        for lo, hi in zip(self.mins, self.maxs):
            prod *= max(0.0, hi - lo)
        return prod

    def margin(self) -> float:
        """Return the sum of edge lengths (useful for split heuristics)."""
        return sum(max(0.0, hi - lo) for lo, hi in zip(self.mins, self.maxs))


@dataclass(frozen=True, slots=True)
class BoxEntry:
    """
    Payload stored in the tree leaf. `obj_id` is an opaque identifier
    (could be an int, str, or any hashable) that the caller can map back to
    their original data.
    """
    obj_id: object
    box: BoundingBox
