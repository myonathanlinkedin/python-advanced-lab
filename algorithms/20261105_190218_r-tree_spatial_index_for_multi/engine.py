from __future__ import annotations
from typing import List, Tuple, Any, Optional, Iterable
from custom_types import Rectangle


class RTreeNode:
    """
    Internal representation of a node in an R‑Tree.
    For leaf nodes, `child` holds the stored data object.
    For internal nodes, `child` holds another RTreeNode.
    """
    __slots__ = ("is_leaf", "entries", "mbr")

    def __init__(self, is_leaf: bool) -> None:
        self.is_leaf: bool = is_leaf
        self.entries: List[Tuple[Rectangle, Any]] = []  # (MBR, child)
        self.mbr: Optional[Rectangle] = None  # cached minimal bounding rectangle

    def add_entry(self, rect: Rectangle, child: Any) -> None:
        self.entries.append((rect, child))
        self.mbr = rect if self.mbr is None else self.mbr.union(rect)

    def recalc_mbr(self) -> None:
        """Recompute the node's MBR from its entries."""
        if not self.entries:
            self.mbr = None
        else:
            it = iter(self.entries)
            first_rect, _ = next(it)
            mbr = first_rect
            for rect, _ in it:
                mbr = mbr.union(rect)
            self.mbr = mbr

    def __repr__(self) -> str:
        typ = "Leaf" if self.is_leaf else "Internal"
        return f"<{typ}Node entries={len(self.entries)} mbr={self.mbr}>"

    # Helper methods used by the RTree algorithm
    def is_overflow(self, max_entries: int) -> bool:
        return len(self.entries) > max_entries


class RTree:
    """
    A deterministic, linear‑split R‑Tree implementation for N‑dimensional rectangles.
    """
    def __init__(self, max_entries: int = 8, min_entries: Optional[int] = None) -> None:
        if max_entries < 2:
            raise ValueError("max_entries must be at least 2")
        self.max_entries: int = max_entries
        self.min_entries: int = min_entries if min_entries is not None else max_entries // 2
        if not (1 <= self.min_entries <= self.max_entries):
            raise ValueError("Invalid min_entries")
        self.root: Optional[RTreeNode] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def insert(self, item: Any, rect: Rectangle) -> None:
        """Insert `item` associated with `rect` into the tree."""
        if self.root is None:
            leaf = RTreeNode(is_leaf=True)
            leaf.add_entry(rect, item)
            self.root = leaf
            return

        split_node = self._insert(self.root, rect, item)
        if split_node is not None:
            # Root overflowed: create new root
            new_root = RTreeNode(is_leaf=False)
            new_root.add_entry(self.root.mbr, self.root)
            new_root.add_entry(split_node.mbr, split_node)
            self.root = new_root

    def search(self, query: Rectangle) -> List[Any]:
        """Return a list of items whose rectangles intersect `query`."""
        if self.root is None:
            return []
        result: List[Any] = []
        self._search(self.root, query, result)
        return result

    # -------------------------------------------------------------------------
    # Internal recursive helpers
    # -------------------------------------------------------------------------
    def _insert(self, node: RTreeNode, rect: Rectangle, item: Any) -> Optional[RTreeNode]:
        """
        Insert entry into `node`. Returns a new node if a split occurs,
        otherwise None.
        """
        if node.is_leaf:
            node.add_entry(rect, item)
            if node.is_overflow(self.max_entries):
                return self._split_node(node)
            else:
                node.recalc_mbr()
                return None
        else:
            # Choose the best child to descend into
            child_node = self._choose_subtree(node, rect)
            split_child = self._insert(child_node, rect, item)
            if split_child is None:
                # No split below; just adjust MBR
                node.recalc_mbr()
                return None
            else:
                # Child split: replace child's entry with two new entries
                # Remove old entry
                for i, (r, c) in enumerate(node.entries):
                    if c is child_node:
                        del node.entries[i]
                        break
                # Add the two new children
                node.add_entry(child_node.mbr, child_node)
                node.add_entry(split_child.mbr, split_child)
                if node.is_overflow(self.max_entries):
                    return self._split_node(node)
                else:
                    node.recalc_mbr()
                    return None

    def _choose_subtree(self, node: RTreeNode, rect: Rectangle) -> RTreeNode:
        """
        For an internal node, pick the child whose MBR needs the least enlargement
        to accommodate `rect`. Ties are broken by smallest area, then by fewest entries.
        """
        best_child = None
        best_enlargement = None
        best_area = None
        best_entry_count = None

        for child_rect, child_node in node.entries:
            enlargement = child_rect.enlargement_needed(rect)
            area = child_rect.area()
            entry_count = len(child_node.entries)

            if (best_enlargement is None or enlargement < best_enlargement or
                (enlargement == best_enlargement and area < best_area) or
                (enlargement == best_enlargement and area == best_area and entry_count < best_entry_count)):
                best_child = child_node
                best_enlargement = enlargement
                best_area = area
                best_entry_count = entry_count

        assert best_child is not None  # internal node must have at least one child
        return best_child

    def _split_node(self, node: RTreeNode) -> RTreeNode:
        """
        Perform a linear split (Guttman 1984) on `node`.
        Returns the newly created node (the second group).
        The original `node` is mutated to become the first group.
        """
        # 1. Pick seeds
        seed1_idx, seed2_idx = self._pick_seeds(node.entries)
        entry1 = node.entries[seed1_idx]
        entry2 = node.entries[seed2_idx]

        # Initialize two groups
        group1 = RTreeNode(is_leaf=node.is_leaf)
        group2 = RTreeNode(is_leaf=node.is_leaf)
        group1.add_entry(*entry1)
        group2.add_entry(*entry2)

        # Remove seeds from the original list
        remaining = [e for i, e in enumerate(node.entries) if i not in (seed1_idx, seed2_idx)]

        # 2. Distribute remaining entries
        while remaining:
            # If one group can take all remaining entries without violating min_entries, assign them.
            if (len(group1.entries) + len(remaining)) == self.min_entries:
                for e in remaining:
                    group1.add_entry(*e)
                break
            if (len(group2.entries) + len(remaining)) == self.min_entries:
                for e in remaining:
                    group2.add_entry(*e)
                break

            # Choose next entry to assign
            entry_idx = self._pick_next(remaining, group1.mbr, group2.mbr)
            rect, child = remaining.pop(entry_idx)

            # Compute enlargement for each group
            inc1 = group1.mbr.enlargement_needed(rect) if group1.mbr else 0.0
            inc2 = group2.mbr.enlargement_needed(rect) if group2.mbr else 0.0

            # Choose group
            if inc1 < inc2:
                group1.add_entry(rect, child)
            elif inc2 < inc1:
                group2.add_entry(rect, child)
            else:
                # Tie‑breaker: smaller area
                area1 = group1.mbr.area() if group1.mbr else 0.0
                area2 = group2.mbr.area() if group2.mbr else 0.0
                if area1 < area2:
                    group1.add_entry(rect, child)
                elif area2 < area1:
                    group2.add_entry(rect, child)
                else:
                    # Final tie‑breaker: fewer entries
                    if len(group1.entries) <= len(group2.entries):
                        group1.add_entry(rect, child)
                    else:
                        group2.add_entry(rect, child)

        # Replace original node's entries with group1's entries
        node.entries = group1.entries
        node.recalc_mbr()
        # Return group2 as the split node
        group2.recalc_mbr()
        return group2

    def _pick_seeds(self, entries: List[Tuple[Rectangle, Any]]) -> Tuple[int, int]:
        """
        Linear seed selection: for each dimension find the pair with the greatest
        normalized separation. Return the indices of the chosen pair.
        """
        dim = entries[0][0].dim
        best_sep = -1.0
        best_pair = (0, 1)  # default fallback

        for d in range(dim):
            low_vals = [(e[0].min_coords[d], i) for i, e in enumerate(entries)]
            high_vals = [(e[0].max_coords[d], i) for i, e in enumerate(entries)]
            low_min, low_min_idx = min(low_vals, key=lambda x: x[0])
            high_max, high_max_idx = max(high_vals, key=lambda x: x[0])

            if low_min_idx == high_max_idx:
                # All entries have identical interval on this dimension; skip.
                continue

            # Compute the width of the whole set on this dimension
            overall_low = min(v for v, _ in low_vals)
            overall_high = max(v for v, _ in high_vals)
            width = overall_high - overall_low
            if width == 0:
                sep = 0.0
            else:
                sep = (high_max - low_min) / width

            if sep > best_sep:
                best_sep = sep
                best_pair = (low_min_idx, high_max_idx)

        return best_pair

    def _pick_next(self, remaining: List[Tuple[Rectangle, Any]],
                   mbr1: Rectangle, mbr2: Rectangle) -> int:
        """
        Choose the entry whose difference in enlargement between the two groups
        is maximal. Returns the index in `remaining`.
        """
        max_diff = -1.0
        chosen_idx = 0
        for idx, (rect, _) in enumerate(remaining):
            inc1 = mbr1.enlargement_needed(rect) if mbr1 else 0.0
            inc2 = mbr2.enlargement_needed(rect) if mbr2 else 0.0
            diff = abs(inc1 - inc2)
            if diff > max_diff:
                max_diff = diff
                chosen_idx = idx
        return chosen_idx

    def _search(self, node: RTreeNode, query: Rectangle, result: List[Any]) -> None:
        """Recursive range query."""
        for entry_rect, child in node.entries:
            if entry_rect.intersects(query):
                if node.is_leaf:
                    result.append(child)
                else:
                    self._search(child, query, result)
