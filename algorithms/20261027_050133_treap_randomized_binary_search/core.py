from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Optional, Tuple, List, Any


@dataclass
class Node:
    key: int
    priority: float
    left: Optional[Node] = None
    right: Optional[Node] = None


class Treap:
    """
    Randomized Binary Search Tree (Treap) with heap priorities.
    Supports insert, delete, search, split, merge, and inorder traversal.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.root: Optional[Node] = None
        self._rng = random.Random(seed)

    # ---------- Public API ----------
    def insert(self, key: int) -> None:
        """Insert a key into the treap. Raises ValueError if key already exists."""
        if self.search(key):
            raise ValueError(f"Key {key} already exists.")
        self.root = self._insert(self.root, Node(key, self._rng.random()))

    def delete(self, key: int) -> None:
        """Delete a key from the treap. Raises KeyError if key not found."""
        if not self.search(key):
            raise KeyError(f"Key {key} not found.")
        self.root = self._delete(self.root, key)

    def search(self, key: int) -> bool:
        """Return True if key exists in the treap."""
        return self._search(self.root, key)

    def inorder(self) -> List[int]:
        """Return a sorted list of keys."""
        return self._inorder(self.root)

    def size(self) -> int:
        """Return number of nodes in the treap."""
        return self._size(self.root)

    # ---------- Internal helpers ----------
    @staticmethod
    def _merge(left: Optional[Node], right: Optional[Node]) -> Optional[Node]:
        if left is None:
            return right
        if right is None:
            return left
        if left.priority > right.priority:
            left.right = Treap._merge(left.right, right)
            return left
        else:
            right.left = Treap._merge(left, right.left)
            return right

    @staticmethod
    def _split(root: Optional[Node], key: int) -> Tuple[Optional[Node], Optional[Node]]:
        if root is None:
            return None, None
        if key <= root.key:
            left, right = Treap._split(root.left, key)
            root.left = right
            return left, root
        else:
            left, right = Treap._split(root.right, key)
            root.right = left
            return root, right

    @staticmethod
    def _insert(root: Optional[Node], node: Node) -> Node:
        if root is None:
            return node
        if node.key < root.key:
            root.left = Treap._insert(root.left, node)
            if root.left.priority > root.priority:
                root = Treap._rotate_right(root)
        else:
            root.right = Treap._insert(root.right, node)
            if root.right.priority > root.priority:
                root = Treap._rotate_left(root)
        return root

    @staticmethod
    def _delete(root: Optional[Node], key: int) -> Optional[Node]:
        if root is None:
            return None
        if key < root.key:
            root.left = Treap._delete(root.left, key)
        elif key > root.key:
            root.right = Treap._delete(root.right, key)
        else:
            # Node to delete found
            return Treap._merge(root.left, root.right)
        return root

    @staticmethod
    def _search(root: Optional[Node], key: int) -> bool:
        while root is not None:
            if key == root.key:
                return True
            root = root.left if key < root.key else root.right
        return False

    @staticmethod
    def _inorder(root: Optional[Node]) -> List[int]:
        if root is None:
            return []
        return Treap._inorder(root.left) + [root.key] + Treap._inorder(root.right)

    @staticmethod
    def _size(root: Optional[Node]) -> int:
        if root is None:
            return 0
        return 1 + Treap._size(root.left) + Treap._size(root.right)

    @staticmethod
    def _rotate_left(root: Node) -> Node:
        new_root = root.right
        root.right = new_root.left
        new_root.left = root
        return new_root

    @staticmethod
    def _rotate_right(root: Node) -> Node:
        new_root = root.left
        root.left = new_root.right
        new_root.right = root
        return new_root

    # ---------- Split & Merge as public methods ----------
    def split(self, key: int) -> Tuple['Treap', 'Treap']:
        """Split treap into two treaps: left with keys < key, right with keys >= key."""
        left_root, right_root = self._split(self.root, key)
        left = Treap()
        right = Treap()
        left.root = left_root
        right.root = right_root
        return left, right

    @staticmethod
    def merge(left: 'Treap', right: 'Treap') -> 'Treap':
        """Merge two treaps into one. Assumes all keys in left < keys in right."""
        merged = Treap()
        merged.root = Treap._merge(left.root, right.root)
        return merged
