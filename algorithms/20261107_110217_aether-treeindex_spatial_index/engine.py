from __future__ import annotations
from typing import List, Any, Tuple, Optional, Sequence
from .types import BoundingBox, LeafEntry, Node


class AetherTreeIndex:
    """
    A simple R‑tree‑like spatial index for N‑dimensional axis aligned bounding boxes.
    The implementation follows the classic linear split algorithm for clarity and
    deterministic behaviour.
    """

    def __init__(self, dimensions: int, max_entries: int = 8, min_entries: Optional[int] = None):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if max_entries < 2:
            raise ValueError("max_entries must be at least 2")
        self.dim = dimensions
        self.max_entries = max_entries
        self.min_entries = min_entries if min_entries is not None else max_entries // 2
        if self.min_entries < 1:
            self.min_entries = 1
        self.root = Node(leaf=True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def insert(self, bbox: BoundingBox, obj: Any) -> None:
        """Insert ``obj`` with its associated ``bbox`` into the tree."""
        if bbox.dim != self.dim:
            raise ValueError("Bounding box dimensionality mismatch with tree")
        leaf = self._choose_leaf(self.root, bbox)
        leaf.add_entry(LeafEntry(bbox, obj))
        self._adjust_tree(leaf)

    def search(self, query: BoundingBox) -> List[Any]:
        """Return a list of objects whose bounding boxes intersect ``query``."""
        if query.dim != self.dim:
            raise ValueError("Query dimensionality mismatch with tree")
        result: List[Any] = []
        self._search_recursive(self.root, query, result)
        return result

    # -------------------------------------------------------------------------
    # Core algorithms
    # -------------------------------------------------------------------------

    def _choose_leaf(self, node: Node, bbox: BoundingBox) -> Node:
        """Descend the tree, picking the child that needs the least enlargement."""
        while not node.leaf:
            best_child = None
            best_enlargement = None
            best_area = None
            for child in node.children:
                current = child.bbox
                enlarged = current.enlarge_to_include(bbox)
                enlargement = enlarged.area() - current.area()
                if (best_enlargement is None or enlargement < best_enlargement or
                    (enlargement == best_enlargement and enlarged.area() < best_area)):
                    best_child = child
                    best_enlargement = enlargement
                    best_area = enlarged.area()
            assert best_child is not None  # tree invariant
            node = best_child
        return node

    def _adjust_tree(self, node: Node) -> None:
        """Propagate splits and bounding‑box updates up to the root."""
        while node is not None:
            if node.is_overflow(self.max_entries):
                node1, node2 = self._split_node(node)
                if node.parent is None:
                    # Create a new root
                    new_root = Node(leaf=False)
                    new_root.add_child(node1)
                    new_root.add_child(node2)
                    self.root = new_root
                    node = new_root
                else:
                    # Replace node with node1 and add node2
                    parent = node.parent
                    # Remove the old reference
                    parent.children.remove(node)
                    parent.add_child(node1)
                    parent.add_child(node2)
                    node = parent
            else:
                node.recalc_bbox()
                node = node.parent

    def _split_node(self, node: Node) -> Tuple[Node, Node]:
        """
        Linear split: pick two seed entries that are farthest apart on any dimension,
        then distribute the remaining entries.
        """
        # Gather all items (entries or children) into a uniform list
        items = node.entries if node.leaf else node.children
        # Choose seeds
        seed1_idx, seed2_idx = self._pick_seeds(items, node.leaf)
        # Initialise two new nodes
        node1 = Node(leaf=node.leaf)
        node2 = Node(leaf=node.leaf)

        # Helper to add an item to a node
        def add_to(node_target: Node, item):
            if node.leaf:
                node_target.add_entry(item)  # type: ignore[arg-type]
            else:
                node_target.add_child(item)  # type: ignore[arg-type]

        # Distribute seeds
        add_to(node1, items[seed1_idx])
        add_to(node2, items[seed2_idx])

        # Remove seeds from the pool
        remaining = [itm for idx, itm in enumerate(items) if idx not in (seed1_idx, seed2_idx)]

        # Distribute remaining items
        while remaining:
            # If one group would fall below min_entries, dump all remaining there
            if (len(node1.entries) if node.leaf else len(node1.children)) + len(remaining) == self.min_entries:
                for itm in remaining:
                    add_to(node1, itm)
                break
            if (len(node2.entries) if node.leaf else len(node2.children)) + len(remaining) == self.min_entries:
                for itm in remaining:
                    add_to(node2, itm)
                break

            # Choose next item based on greatest difference in enlargement
            best_idx = None
            best_diff = None
            best_target = None  # 1 or 2
            for idx, itm in enumerate(remaining):
                bbox_itm = itm.bbox if node.leaf else itm.bbox
                # Enlargement if added to node1
                e1 = node1.bbox.enlarge_to_include(bbox_itm).area() - node1.bbox.area() if node1.bbox else 0.0
                e2 = node2.bbox.enlarge_to_include(bbox_itm).area() - node2.bbox.area() if node2.bbox else 0.0
                diff = abs(e1 - e2)
                if best_diff is None or diff > best_diff:
                    best_diff = diff
                    best_idx = idx
                    best_target = 1 if e1 < e2 else 2

            # Assign the chosen item
            chosen_item = remaining.pop(best_idx)  # type: ignore[arg-type]
            if best_target == 1:
                add_to(node1, chosen_item)
            else:
                add_to(node2, chosen_item)

        node1.recalc_bbox()
        node2.recalc_bbox()
        return node1, node2

    def _pick_seeds(self, items: List[Any], leaf: bool) -> Tuple[int, int]:
        """
        Linear seed selection: for each dimension compute the normalized separation
        between the lowest low and highest high. Choose the pair with the greatest
        separation.
        """
        dim = self.dim
        low_high = [(itm.bbox.mins[i], itm.bbox.maxs[i]) for itm in items for i in range(dim)]
        # We'll compute for each dimension the pair with max separation
        best_sep = -1.0
        best_pair = (0, 1)
        for d in range(dim):
            # Find the item with the lowest low and the item with the highest high
            low_idx = high_idx = None
            low_val = float('inf')
            high_val = float('-inf')
            for idx, itm in enumerate(items):
                low = itm.bbox.mins[d]
                high = itm.bbox.maxs[d]
                if low < low_val:
                    low_val = low
                    low_idx = idx
                if high > high_val:
                    high_val = high
                    high_idx = idx
            if low_idx is None or high_idx is None:
                continue
            width = max(itm.bbox.maxs[d] for itm in items) - min(itm.bbox.mins[d] for itm in items)
            if width == 0:
                sep = 0.0
            else:
                sep = (high_val - low_val) / width
            if sep > best_sep:
                best_sep = sep
                best_pair = (low_idx, high_idx)
        # Ensure distinct indices
        if best_pair[0] == best_pair[1]:
            # fallback: first two items
            return 0, 1
        return best_pair

    def _search_recursive(self, node: Node, query: BoundingBox, result: List[Any]) -> None:
        """Depth‑first search collecting payloads intersecting ``query``."""
        if node.bbox is None or not node.bbox.intersects(query):
            return
        if node.leaf:
            for entry in node.entries:
                if entry.bbox.intersects(query):
                    result.append(entry.obj)
        else:
            for child in node.children:
                self._search_recursive(child, query, result)
