from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Any


@dataclass(frozen=True, slots=True)
class Rectangle:
    """
    Axis-aligned bounding box in N-dimensional space.
    Stored as two tuples of equal length:
        min_coords[i] <= max_coords[i] for each dimension i.
    """
    min_coords: Tuple[float, ...]
    max_coords: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.min_coords) != len(self.max_coords):
            raise ValueError("min_coords and max_coords must have the same dimension")
        for lo, hi in zip(self.min_coords, self.max_coords):
            if lo > hi:
                raise ValueError("Each min coordinate must be <= the corresponding max coordinate")

    @property
    def dim(self) -> int:
        return len(self.min_coords)

    def area(self) -> float:
        """Hyper‑volume of the rectangle."""
        prod = 1.0
        for lo, hi in zip(self.min_coords, self.max_coords):
            prod *= (hi - lo)
        return prod

    def union(self, other: Rectangle) -> Rectangle:
        """Return the minimal rectangle containing both self and other."""
        if self.dim != other.dim:
            raise ValueError("Rectangles must have the same dimension to compute union")
        new_min = tuple(min(a, b) for a, b in zip(self.min_coords, other.min_coords))
        new_max = tuple(max(a, b) for a, b in zip(self.max_coords, other.max_coords))
        return Rectangle(new_min, new_max)

    def enlargement_needed(self, other: Rectangle) -> float:
        """Additional area required to enlarge self so it contains other."""
        return self.union(other).area() - self.area()

    def contains_point(self, point: Tuple[float, ...]) -> bool:
        if len(point) != self.dim:
            raise ValueError("Point dimension mismatch")
        return all(lo <= p <= hi for lo, p, hi in zip(self.min_coords, point, self.max_coords))

    def intersects(self, other: Rectangle) -> bool:
        """True if self and other overlap (including touching edges)."""
        if self.dim != other.dim:
            raise ValueError("Rectangles must have the same dimension to test intersection")
        return all(not (self.max_coords[i] < other.min_coords[i] or other.max_coords[i] < self.min_coords[i])
                   for i in range(self.dim))

    def __repr__(self) -> str:
        return f"Rectangle(min={self.min_coords}, max={self.max_coords})"
