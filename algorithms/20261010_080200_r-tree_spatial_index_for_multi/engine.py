from __future__ import annotations
from typing import List, Any, Tuple, Optional, Union
from .types import Rectangle, RTreeNode


class RTree:
    """A simple, deterministic R‑Tree implementation using the linear split algorithm."""
    def __init__(self, max_entries: int = 4) -> None:
        if max_entries < 2:
            raise ValueError("max_entries must be at least 2.")
        self.max_entries = max_entries
        self.min_entries = max_entries // 2
        self.root = RTreeNode(leaf=True)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def insert(self, rect: Rectangle, obj: Any) -> None:
        leaf = self._choose_leaf(self.root, rect)
        leaf.add_entry(rect, obj)
        self._adjust_tree(leaf)

    def search(self, rect: Rectangle) -> List[Any]:
        """Return all objects whose rectangles intersect `rect`."""
        result: List[Any] = []
        self._search_recursive(self.root, rect, result)
        return result

    def __len__(self) -> int:
        return self._count_entries(self.root)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _choose_leaf(self, node: RTreeNode, rect: Rectangle) -> RTreeNode:
        """Select the leaf node in which to place a new entry."""
        if node.leaf:
            return node
        # Choose the child requiring the least enlargement; tie‑break by smallest area.
        best_child = None
        best_enlargement = float('inf')
        best_area = float('inf')
        for child_rect, child_node in node.entries:
            assert isinstance(child_node, RTreeNode)
            enlargement = child_rect.enlargement_needed(rect)
            area = child_rect.area()
            if enlargement < best_enlargement or (enlargement == best_enlargement and area < best_area):
                best_child = child_node
                best_enlargement = enlargement
                best_area = area
        assert best_child is not None
        return self._choose_leaf(best_child, rect)

    def _adjust_tree(self, node: RTreeNode) -> None:
        """Propagate node splits and MBR updates up to the root."""
        while node is not None:
            if node.is_overflow(self.max_entries):
                node1, node2 = self._split_node(node)
                if node.parent is None:
                    # Root overflow – create new root.
                    new_root = RTreeNode(leaf=False)
                    new_root.add_entry(node1.mbr(), node1)
                    new_root.add_entry(node2.mbr(), node2)
                    self.root = new_root
                    node = new_root
                else:
                    # Replace node in parent with node1, and add node2.
                    parent = node.parent
                    # Remove old entry for the overflowing node.
                    for rect, child in parent.entries:
                        if child is node:
                            parent.remove_entry(rect, child)
                            break
                    parent.add_entry(node1.mbr(), node1)
                    parent.add_entry(node2.mbr(), node2)
                    node = parent
            else:
                # No overflow – just tighten MBRs.
                if node.parent is not None:
                    # Update the entry rectangle in the parent.
                    parent = node.parent
                    for idx, (rect, child) in enumerate(parent.entries):
                        if child is node:
                            parent.entries[idx] = (node.mbr(), node)
                            break
                node = node.parent

    def _split_node(self, node: RTreeNode) -> Tuple[RTreeNode, RTreeNode]:
        """Linear split as described by Guttman."""
        # 1. Pick two seed entries that are farthest apart on any dimension.
        seed1_idx, seed2_idx = self._pick_seeds(node.entries)
        entry1 = node.entries[seed1_idx]
        entry2 = node.entries[seed2_idx]

        # Initialize two groups.
        group1 = RTreeNode(leaf=node.leaf)
        group2 = RTreeNode(leaf=node.leaf)
        group1.add_entry(*entry1)
        group2.add_entry(*entry2)

        # Remove seeds from the original list.
        remaining = [e for i, e in enumerate(node.entries) if i not in (seed1_idx, seed2_idx)]

        # 2. Distribute remaining entries.
        while remaining:
            # If one group would contain the minimum number of entries after adding all remaining,
            # assign all remaining to that group.
            if (len(group1.entries) + len(remaining)) == self.min_entries:
                for e in remaining:
                    group1.add_entry(*e)
                break
            if (len(group2.entries) + len(remaining)) == self.min_entries:
                for e in remaining:
                    group2.add_entry(*e)
                break

            # Choose the next entry to assign.
            best_idx = None
            best_diff = -1.0
            best_group = None  # 1 or 2
            for idx, (rect, obj) in enumerate(remaining):
                d1 = group1.mbr().enlargement_needed(rect)
                d2 = group2.mbr().enlargement_needed(rect)
                diff = abs(d1 - d2)
                if diff > best_diff:
                    best_diff = diff
                    best_idx = idx
                    best_group = 1 if d1 < d2 else 2
                elif diff == best_diff:
                    # Tie‑breaker: choose the group with smaller area.
                    area1 = group1.mbr().area()
                    area2 = group2.mbr().area()
                    if area1 < area2:
                        best_group = 1
                    elif area2 < area1:
                        best_group = 2
                    else:
                        # Final tie‑breaker: fewer entries.
                        best_group = 1 if len(group1.entries) <= len(group2.entries) else 2

            rect, obj = remaining.pop(best_idx)  # type: ignore
            if best_group == 1:
                group1.add_entry(rect, obj)
            else:
                group2.add_entry(rect, obj)

        return group1, group2

    def _pick_seeds(self, entries: List[Tuple[Rectangle, Union[Any, RTreeNode]]]) -> Tuple[int, int]:
        """Select two seeds that would waste the most space if put together."""
        dim = entries[0][0].dim
        highest_waste = -1.0
        seed1 = seed2 = -1
        for d in range(dim):
            low = min(e[0].min_corner[d] for e in entries)
            high = max(e[0].max_corner[d] for e in entries)
            width = high - low
            if width == 0:
                continue
            # Normalized separation.
            sep = (max(e[0].max_corner[d] for e in entries) - min(e[0].min_corner[d] for e in entries)) / width
            # Find entries with extreme low and high.
            low_entry = min(range(len(entries)), key=lambda i: entries[i][0].min_corner[d])
            high_entry = max(range(len(entries)), key=lambda i: entries[i][0].max_corner[d])
            waste = abs(sep)
            if waste > highest_waste:
                highest_waste = waste
                seed1, seed2 = low_entry, high_entry
        if seed1 == seed2:
            # Fallback: pick first two distinct entries.
            seed1, seed2 = 0, 1
        return seed1, seed2

    def _search_recursive(self, node: RTreeNode, rect: Rectangle, out: List[Any]) -> None:
        for entry_rect, obj in node.entries:
            if entry_rect.intersects(rect):
                if node.leaf:
                    out.append(obj)
                else:
                    assert isinstance(obj, RTreeNode)
                    self._search_recursive(obj, rect, out)

    def _count_entries(self, node: RTreeNode) -> int:
        if node.leaf:
            return len(node.entries)
        return sum(self._count_entries(child) for _, child in node.entries)
