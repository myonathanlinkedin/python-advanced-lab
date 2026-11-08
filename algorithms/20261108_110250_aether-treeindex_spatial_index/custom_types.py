"""
Domain models and memory structures for the Aether-TreeIndex.

This module defines the core data structures:
- Point: A point in N-dimensional space.
- BoundingBox: An axis-aligned bounding box (AABB) in N-dimensional space.
- Node: A node in the Aether-Tree, containing either a leaf (with items) or internal children.
- AetherTreeIndex: The top-level spatial index structure.

Design invariants:
- All coordinates are floats for generality.
- BoundingBoxes are canonical: min_coord[i] <= max_coord[i] for all i.
- The tree is a quadtree/octree generalization (M-ary tree) with configurable branching factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass(frozen=True)
class Point:
    """An immutable point in N-dimensional Euclidean space."""
    coords: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.coords:
            raise ValueError("Point must have at least one dimension.")

    @property
    def dim(self) -> int:
        return len(self.coords)

    def distance_squared(self, other: Point) -> float:
        """Euclidean distance squared to another point. O(d)."""
        if self.dim != other.dim:
            raise ValueError("Dimension mismatch.")
        return sum((a - b) ** 2 for a, b in zip(self.coords, other.coords))

    def distance(self, other: Point) -> float:
        """Euclidean distance to another point. O(d)."""
        return self.distance_squared(other) ** 0.5

    def __repr__(self) -> str:
        return f"Point({self.coords})"


@dataclass
class BoundingBox:
    """
    An axis-aligned bounding box in N-dimensional space.
    Canonical form: min_coord[i] <= max_coord[i] for all i.
    """
    min_coord: Tuple[float, ...]
    max_coord: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.min_coord) != len(self.max_coord):
            raise ValueError("min_coord and max_coord must have same dimension.")
        # Enforce canonical ordering
        self.min_coord = tuple(min(a, b) for a, b in zip(self.min_coord, self.max_coord))
        self.max_coord = tuple(max(a, b) for a, b in zip(self.min_coord, self.max_coord))

    @property
    def dim(self) -> int:
        return len(self.min_coord)

    @staticmethod
    def from_point(p: Point) -> BoundingBox:
        """Create a degenerate bounding box (zero volume) at a point."""
        return BoundingBox(min_coord=p.coords, max_coord=p.coords)

    @staticmethod
    def from_points(points: List[Point]) -> BoundingBox:
        """Create the minimal bounding box enclosing all given points."""
        if not points:
            raise ValueError("Cannot create bounding box from empty list.")
        dim = points[0].dim
        mins = [float('inf')] * dim
        maxs = [float('-inf')] * dim
        for p in points:
            if p.dim != dim:
                raise ValueError("All points must have same dimension.")
            for i in range(dim):
                if p.coords[i] < mins[i]:
                    mins[i] = p.coords[i]
                if p.coords[i] > maxs[i]:
                    maxs[i] = p.coords[i]
        return BoundingBox(min_coord=tuple(mins), max_coord=tuple(maxs))

    def contains_point(self, p: Point) -> bool:
        """Check if a point lies within this bounding box (inclusive). O(d)."""
        if p.dim != self.dim:
            return False
        return all(self.min_coord[i] <= p.coords[i] <= self.max_coord[i] for i in range(self.dim))

    def intersects(self, other: BoundingBox) -> bool:
        """Check if two bounding boxes intersect (inclusive). O(d)."""
        if self.dim != other.dim:
            return False
        for i in range(self.dim):
            if self.max_coord[i] < other.min_coord[i] or other.max_coord[i] < self.min_coord[i]:
                return False
        return True

    def union(self, other: BoundingBox) -> BoundingBox:
        """Return the minimal bounding box enclosing both boxes. O(d)."""
        if self.dim != other.dim:
            raise ValueError("Dimension mismatch.")
        mins = tuple(min(a, b) for a, b in zip(self.min_coord, other.min_coord))
        maxs = tuple(max(a, b) for a, b in zip(self.max_coord, other.max_coord))
        return BoundingBox(min_coord=mins, max_coord=maxs)

    def volume(self) -> float:
        """Compute the N-dimensional volume (hyper-volume). O(d)."""
        vol = 1.0
        for i in range(self.dim):
            vol *= (self.max_coord[i] - self.min_coord[i])
        return vol

    def center(self) -> Point:
        """Return the center point of the bounding box. O(d)."""
        return Point(coords=tuple((a + b) / 2.0 for a, b in zip(self.min_coord, max_coord)))

    def distance_to_point(self, p: Point) -> float:
        """
        Minimum Euclidean distance from a point to this bounding box. O(d).
        If point is inside, distance is 0.
        """
        if p.dim != self.dim:
            raise ValueError("Dimension mismatch.")
        dist_sq = 0.0
        for i in range(self.dim):
            if p.coords[i] < self.min_coord[i]:
                dist_sq += (self.min_coord[i] - p.coords[i]) ** 2
            elif p.coords[i] > self.max_coord[i]:
                dist_sq += (p.coords[i] - self.max_coord[i]) ** 2
        return dist_sq ** 0.5

    def distance_to_box(self, other: BoundingBox) -> float:
        """
        Minimum Euclidean distance between two bounding boxes. O(d).
        If they intersect, distance is 0.
        """
        if self.dim != other.dim:
            raise ValueError("Dimension mismatch.")
        dist_sq = 0.0
        for i in range(self.dim):
            if self.max_coord[i] < other.min_coord[i]:
                dist_sq += (other.min_coord[i] - self.max_coord[i]) ** 2
            elif other.max_coord[i] < self.min_coord[i]:
                dist_sq += (self.min_coord[i] - other.max_coord[i]) ** 2
        return dist_sq ** 0.5

    def __repr__(self) -> str:
        return f"BoundingBox(min={self.min_coord}, max={self.max_coord})"


@dataclass
class Item:
    """
    A spatial item stored in the Aether-Tree.
    Each item has a bounding box and an associated payload.
    """
    bbox: BoundingBox
    payload: Any

    def __repr__(self) -> str:
        return f"Item(bbox={self.bbox}, payload={self.payload!r})"


class Node:
    """
    A node in the Aether-Tree.
    
    - If is_leaf is True: contains a list of Items (up to max_items).
    - If is_leaf is False: contains a list of child Nodes (exactly branching_factor children).
    
    Each node stores its bounding box (the union of all items/children in its subtree).
    """

    def __init__(self, bbox: BoundingBox, is_leaf: bool = True,
                 max_items: int = 4, branching_factor: int = 4) -> None:
        self.bbox = bbox
        self.is_leaf = is_leaf
        self.max_items = max_items
        self.branching_factor = branching_factor
        self.items: List[Item] = []
        self.children: List[Node] = []

    def add_item(self, item: Item) -> None:
        """Add an item to this leaf node and update the bounding box."""
        self.items.append(item)
        self.bbox = self.bbox.union(item.bbox)

    def add_child(self, child: Node) -> None:
        """Add a child node and update the bounding box."""
        self.children.append(child)
        self.bbox = self.bbox.union(child.bbox)

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"LeafNode(bbox={self.bbox}, items={len(self.items)})"
        return f"InternalNode(bbox={self.bbox}, children={len(self.children)})"


class AetherTreeIndex:
    """
    The Aether-TreeIndex: a multidimensional spatial index for bounding boxes.
    
    This is a generalization of R-trees and quadtrees/octrees. Key properties:
    - M-ary tree with configurable branching factor.
    - Each leaf holds up to max_items items.
    - Internal nodes partition space by splitting along the dimension with the largest spread.
    - Supports: insert, range query, nearest-neighbor query, and bulk load.
    
    Time complexity:
    - Insert: O(log n) average, O(n) worst case (degenerate splits).
    - Range query: O(sqrt(n) + k) average for d-dimensional data, where k is result count.
    - Nearest neighbor: O(log n) average with pruning.
    
    Space complexity: O(n) for n items.
    """

    def __init__(self, dim: int, max_items: int = 4, branching_factor: int = 4) -> None:
        if dim < 1:
            raise ValueError("Dimension must be >= 1.")
        if max_items < 1:
            raise ValueError("max_items must be >= 1.")
        if branching_factor < 2:
            raise ValueError("branching_factor must be >= 2.")
        self.dim = dim
        self.max_items = max_items
        self.branching_factor = branching_factor
        self.root: Optional[Node] = None
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"AetherTreeIndex(dim={self.dim}, size={self._size}, max_items={self.max_items}, branching_factor={self.branching_factor})"
