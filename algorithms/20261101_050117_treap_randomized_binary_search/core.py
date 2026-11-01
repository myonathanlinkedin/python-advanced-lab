import random
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Callable


@dataclass
class _Node:
    key: Any
    value: Any
    priority: int
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class Treap:
    """
    Randomized binary search tree (Treap) where each node respects:
      - BST property on `key`
      - Heap property on `priority` (max‑heap)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._root: Optional[_Node] = None
        self._rand = random.Random(seed)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _split(self, root: Optional[_Node], key: Any) -> Tuple[Optional[_Node], Optional[_Node]]:
        """Split `root` into (left, right) where left keys < key and right keys >= key."""
        if root is None:
            return None, None
        if key <= root.key:
            left, new_left = self._split(root.left, key)
            root.left = new_left
            return left, root
        else:
            new_right, right = self._split(root.right, key)
            root.right = new_right
            return root, right

    def _merge(self, left: Optional[_Node], right: Optional[_Node]) -> Optional[_Node]:
        """Merge two treaps where all keys in `left` < keys in `right`."""
        if left is None or right is None:
            return left or right
        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            return left
        else:
            right.left = self._merge(left, right.left)
            return right

    def _insert_node(self, root: Optional[_Node], node: _Node) -> _Node:
        """Insert `node` into subtree rooted at `root`."""
        if root is None:
            return node
        if node.priority > root.priority:
            left, right = self._split(root, node.key)
            node.left = left
            node.right = right
            return node
        elif node.key < root.key:
            root.left = self._insert_node(root.left, node)
        elif node.key > root.key:
            root.right = self._insert_node(root.right, node)
        else:  # duplicate key – replace value
            root.value = node.value
        return root

    def _erase(self, root: Optional[_Node], key: Any) -> Optional[_Node]:
        """Erase `key` from subtree rooted at `root`."""
        if root is None:
            return None
        if key < root.key:
            root.left = self._erase(root.left, key)
            return root
        if key > root.key:
            root.right = self._erase(root.right, key)
            return root
        # key == root.key → remove this node
        return self._merge(root.left, root.right)

    def _find(self, root: Optional[_Node], key: Any) -> Optional[_Node]:
        while root is not None:
            if key < root.key:
                root = root.left
            elif key > root.key:
                root = root.right
            else:
                return root
        return None

    def _inorder(self, root: Optional[_Node], out: List[Tuple[Any, Any]]) -> None:
        if root is None:
            return
        self._inorder(root.left, out)
        out.append((root.key, root.value))
        self._inorder(root.right, out)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def insert(self, key: Any, value: Any = None) -> None:
        """Insert `key` with associated `value`. If `key` exists, its value is updated."""
        node = _Node(key=key, value=value, priority=self._rand.getrandbits(30))
        self._root = self._insert_node(self._root, node)

    def erase(self, key: Any) -> None:
        """Remove `key` from the treap. No error if `key` is absent."""
        self._root = self._erase(self._root, key)

    def find(self, key: Any) -> Optional[Any]:
        """Return the value associated with `key`, or None if absent."""
        node = self._find(self._root, key)
        return None if node is None else node.value

    def inorder(self) -> List[Tuple[Any, Any]]:
        """Return a list of (key, value) pairs in sorted order."""
        out: List[Tuple[Any, Any]] = []
        self._inorder(self._root, out)
        return out

    # --------------------------------------------------------------------- #
    # Validation utilities (used by tests)
    # --------------------------------------------------------------------- #
    def _check_bst_property(self, node: Optional[_Node], lo: Any = None, hi: Any = None) -> bool:
        if node is None:
            return True
        if (lo is not None and node.key <= lo) or (hi is not None and node.key >= hi):
            return False
        return (self._check_bst_property(node.left, lo, node.key) and
                self._check_bst_property(node.right, node.key, hi))

    def _check_heap_property(self, node: Optional[_Node]) -> bool:
        if node is None:
            return True
        left_ok = (node.left is None or node.priority >= node.left.priority) and self._check_heap_property(node.left)
        right_ok = (node.right is None or node.priority >= node.right.priority) and self._check_heap_property(node.right)
        return left_ok and right_ok

    def validate(self) -> bool:
        """Return True iff both BST and heap invariants hold."""
        return self._check_bst_property(self._root) and self._check_heap_property(self._root)
