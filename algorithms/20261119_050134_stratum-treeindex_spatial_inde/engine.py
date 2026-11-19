from __future__ import annotations

from typing import List, Optional, Tuple, Union
from custom_types import BoundingBox, BoxEntry


class TreeNode:
    """
    Internal node or leaf of the Stratum‑TreeIndex.
    - If `leaf` is True, `children` holds BoxEntry objects.
    - If `leaf` is False, `children` holds TreeNode objects.
    The node's `bbox` always encloses all its children.
    """
    __slots__ = ("bbox", "children", "leaf", "max_entries", "min_entries")

    def __init__(
        self,
        *,
        leaf: bool,
        max_entries: int = 8,
        min_entries: Optional[int] = None,
        bbox: Optional[BoundingBox] = None,
    ) -> None:
        if max_entries < 2:
            raise ValueError("max_entries must be >= 2")
        self.leaf: bool = leaf
        self.max_entries: int = max_entries
        self.min_entries: int = min_entries if min_entries is not None else max_entries // 2
        if self.min_entries < 1:
            raise ValueError("min_entries must be >= 1")
        self.children: List[Union[TreeNode, BoxEntry]] = []
        self.bbox: Optional[BoundingBox] = bbox

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def _recalculate_bbox(self) -> None:
        """Recompute the node's bounding box from its children."""
        if not self.children:
            self.bbox = None
            return
        it = iter(self.children)
        first = next(it)
        cur_bbox = first.box if isinstance(first, BoxEntry) else first.bbox  # type: ignore
        for child in it:
            child_bbox = child.box if isinstance(child, BoxEntry) else child.bbox  # type: ignore
            cur_bbox = cur_bbox.enlarged(child_bbox)  # type: ignore
        self.bbox = cur_bbox

    # --------------------------------------------------------------------- #
    # Insertion
    # --------------------------------------------------------------------- #
    def insert(self, entry: BoxEntry) -> Optional[TreeNode]:
        """
        Insert `entry` into the subtree rooted at this node.
        Returns a new node if a split occurs, otherwise None.
        """
        if self.leaf:
            self.children.append(entry)
            if self.bbox is None:
                self.bbox = entry.box
            else:
                self.bbox = self.bbox.enlarged(entry.box)
            if len(self.children) > self.max_entries:
                return self._split()
            return None

        # Choose the best child to descend into (least enlargement)
        best_child = self._choose_subtree(entry.box)
        split_node = best_child.insert(entry)
        # After insertion, enlarge this node's bbox if needed
        if self.bbox is None:
            self.bbox = entry.box
        else:
            self.bbox = self.bbox.enlarged(entry.box)

        if split_node:
            # Need to add the new node as a child
            self.children.append(split_node)
            self.bbox = self.bbox.enlarged(split_node.bbox)  # type: ignore
            if len(self.children) > self.max_entries:
                return self._split()
        return None

    def _choose_subtree(self, box: BoundingBox) -> TreeNode:
        """Select the child node that requires the least enlargement to accommodate `box`."""
        best = None
        best_enlargement = None
        best_area = None
        for child in self.children:  # type: ignore
            assert isinstance(child, TreeNode)
            cur_bbox = child.bbox
            assert cur_bbox is not None
            enlarged = cur_bbox.enlarged(box)
            enlargement = enlarged.area() - cur_bbox.area()
            if (best_enlargement is None) or (enlargement < best_enlargement) or (
                enlargement == best_enlargement and cur_bbox.area() < best_area
            ):
                best = child
                best_enlargement = enlargement
                best_area = cur_bbox.area()
        assert best is not None
        return best

    # --------------------------------------------------------------------- #
    # Splitting (Linear split, similar to R‑Tree)
    # --------------------------------------------------------------------- #
    def _split(self) -> TreeNode:
        """
        Perform a linear split of this node's children.
        Returns the newly created sibling node.
        """
        # 1. Pick seeds
        seed1, seed2 = self._pick_seeds()
        group1: List[Union[TreeNode, BoxEntry]] = [seed1]
        group2: List[Union[TreeNode, BoxEntry]] = [seed2]

        # Remove seeds from the pool
        remaining = [c for c in self.children if c not in (seed1, seed2)]

        # Helper to compute bbox of a group
        def group_bbox(group: List[Union[TreeNode, BoxEntry]]) -> BoundingBox:
            it = iter(group)
            first = next(it)
            cur = first.box if isinstance(first, BoxEntry) else first.bbox  # type: ignore
            for child in it:
                child_bbox = child.box if isinstance(child, BoxEntry) else child.bbox  # type: ignore
                cur = cur.enlarged(child_bbox)  # type: ignore
            return cur

        # 2. Distribute remaining entries
        while remaining:
            # Enforce minimum fill
            if (len(group1) + len(remaining)) == self.min_entries:
                group1.extend(remaining)
                break
            if (len(group2) + len(remaining)) == self.min_entries:
                group2.extend(remaining)
                break

            entry = remaining.pop(0)
            bbox1 = group_bbox(group1)
            bbox2 = group_bbox(group2)

            inc1 = bbox1.enlarged(entry.box if isinstance(entry, BoxEntry) else entry.bbox).area() - bbox1.area()
            inc2 = bbox2.enlarged(entry.box if isinstance(entry, BoxEntry) else entry.bbox).area() - bbox2.area()

            if inc1 < inc2:
                group1.append(entry)
            elif inc2 < inc1:
                group2.append(entry)
            else:
                # Tie‑breaker: add to the group with smaller area, then fewer entries
                if bbox1.area() < bbox2.area():
                    group1.append(entry)
                elif bbox2.area() < bbox1.area():
                    group2.append(entry)
                else:
                    if len(group1) <= len(group2):
                        group1.append(entry)
                    else:
                        group2.append(entry)

        # 3. Build new sibling node
        sibling = TreeNode(
            leaf=self.leaf,
            max_entries=self.max_entries,
            min_entries=self.min_entries,
        )
        sibling.children = group2
        sibling._recalculate_bbox()

        # 4. Keep group1 in the current node
        self.children = group1
        self._recalculate_bbox()

        return sibling

    def _pick_seeds(self) -> Tuple[Union[TreeNode, BoxEntry], Union[TreeNode, BoxEntry]]:
        """
        Linear seed selection: for each dimension find the pair with the greatest
        normalized separation and pick the pair with the overall greatest separation.
        """
        dim = self.children[0].box.dim if isinstance(self.children[0], BoxEntry) else self.children[0].bbox.dim  # type: ignore
        highest_norm_sep = -1.0
        seed_pair: Tuple[Union[TreeNode, BoxEntry], Union[TreeNode, BoxEntry]] = (self.children[0], self.children[1])  # type: ignore

        for d in range(dim):
            low = min(
                (c.box.mins[d] if isinstance(c, BoxEntry) else c.bbox.mins[d] for c in self.children),  # type: ignore
                default=0.0,
            )
            high = max(
                (c.box.maxs[d] if isinstance(c, BoxEntry) else c.bbox.maxs[d] for c in self.children),  # type: ignore
                default=0.0,
            )
            width = high - low
            if width == 0:
                continue  # all boxes aligned on this axis

            # Find entries with min low and max high
            min_low_entry = min(
                self.children,
                key=lambda c: c.box.mins[d] if isinstance(c, BoxEntry) else c.bbox.mins[d],  # type: ignore
            )
            max_high_entry = max(
                self.children,
                key=lambda c: c.box.maxs[d] if isinstance(c, BoxEntry) else c.bbox.maxs[d],  # type: ignore
            )
            if min_low_entry is max_high_entry:
                continue  # same entry, cannot be a seed pair for this axis

            sep = (max_high_entry.box.mins[d] if isinstance(max_high_entry, BoxEntry) else max_high_entry.bbox.mins[d]) - \
                  (min_low_entry.box.maxs[d] if isinstance(min_low_entry, BoxEntry) else min_low_entry.bbox.maxs[d])
            norm_sep = sep / width
            if norm_sep > highest_norm_sep:
                highest_norm_sep = norm_sep
                seed_pair = (min_low_entry, max_high_entry)

        return seed_pair

    # --------------------------------------------------------------------- #
    # Query
    # --------------------------------------------------------------------- #
    def query(self, box: BoundingBox) -> List[BoxEntry]:
        """
        Return all leaf entries whose bounding boxes intersect `box`.
        """
        result: List[BoxEntry] = []
        if self.bbox is None or not self.bbox.intersects(box):
            return result

        if self.leaf:
            for entry in self.children:  # type: ignore
                assert isinstance(entry, BoxEntry)
                if entry.box.intersects(box):
                    result.append(entry)
        else:
            for child in self.children:  # type: ignore
                assert isinstance(child, TreeNode)
                result.extend(child.query(box))
        return result


class StratumTreeIndex:
    """
    Public façade for the spatial index.
    """
    __slots__ = ("root", "dimension", "max_entries", "min_entries")

    def __init__(self, dimension: int, max_entries: int = 8, min_entries: Optional[int] = None) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self.dimension = dimension
        self.max_entries = max_entries
        self.min_entries = min_entries
        # Start with an empty leaf node
        self.root = TreeNode(
            leaf=True,
            max_entries=self.max_entries,
            min_entries=self.min_entries,
        )

    def insert(self, obj_id: object, mins: Sequence[float], maxs: Sequence[float]) -> None:
        """Insert a new bounding box associated with `obj_id`."""
        if len(mins) != self.dimension or len(maxs) != self.dimension:
            raise ValueError("Bounding box dimensionality mismatch")
        entry = BoxEntry(obj_id, BoundingBox(tuple(mins), tuple(maxs)))
        split = self.root.insert(entry)
        if split:
            # Grow tree height
            new_root = TreeNode(
                leaf=False,
                max_entries=self.max_entries,
                min_entries=self.min_entries,
            )
            new_root.children = [self.root, split]
            new_root._recalculate_bbox()
            self.root = new_root

    def query(self, mins: Sequence[float], maxs: Sequence[float]) -> List[object]:
        """Return a list of `obj_id`s whose boxes intersect the query box."""
        if len(mins) != self.dimension or len(maxs) != self.dimension:
            raise ValueError("Query box dimensionality mismatch")
        query_box = BoundingBox(tuple(mins), tuple(maxs))
        entries = self.root.query(query_box)
        return [e.obj_id for e in entries]
