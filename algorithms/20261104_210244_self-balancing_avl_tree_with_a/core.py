from __future__ import annotations
from typing import Generic, TypeVar, Optional, Tuple, List

K = TypeVar('K')
V = TypeVar('V')


class AVLNode(Generic[K, V]):
    __slots__ = ('key', 'value', 'left', 'right', 'height')

    def __init__(self, key: K, value: V) -> None:
        self.key: K = key
        self.value: V = value
        self.left: Optional[AVLNode[K, V]] = None
        self.right: Optional[AVLNode[K, V]] = None
        self.height: int = 1  # leaf node height


class AVLTree(Generic[K, V]):
    """Self‑balancing AVL tree."""

    def __init__(self) -> None:
        self.root: Optional[AVLNode[K, V]] = None

    # ---------- Public API ----------
    def insert(self, key: K, value: V) -> None:
        """Insert key/value pair. If key exists, its value is replaced."""
        self.root = self._insert(self.root, key, value)

    def delete(self, key: K) -> None:
        """Remove node with given key. No effect if key not present."""
        self.root = self._delete(self.root, key)

    def find(self, key: K) -> Optional[V]:
        """Return value associated with key, or None if absent."""
        node = self._find(self.root, key)
        return node.value if node else None

    def inorder(self) -> List[Tuple[K, V]]:
        """Return list of (key, value) pairs in sorted order."""
        result: List[Tuple[K, V]] = []
        self._inorder(self.root, result)
        return result

    def is_balanced(self) -> bool:
        """Validate AVL balance property for the whole tree."""
        return self._check_balance(self.root)[1]

    # ---------- Internal helpers ----------
    def _height(self, node: Optional[AVLNode[K, V]]) -> int:
        return node.height if node else 0

    def _update_height(self, node: AVLNode[K, V]) -> None:
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _balance_factor(self, node: AVLNode[K, V]) -> int:
        return self._height(node.left) - self._height(node.right)

    def _rotate_right(self, y: AVLNode[K, V]) -> AVLNode[K, V]:
        x = y.left  # type: ignore
        T2 = x.right  # type: ignore

        # Perform rotation
        x.right = y
        y.left = T2

        # Update heights
        self._update_height(y)
        self._update_height(x)

        return x

    def _rotate_left(self, x: AVLNode[K, V]) -> AVLNode[K, V]:
        y = x.right  # type: ignore
        T2 = y.left  # type: ignore

        # Perform rotation
        y.left = x
        x.right = T2

        # Update heights
        self._update_height(x)
        self._update_height(y)

        return y

    def _rebalance(self, node: AVLNode[K, V]) -> AVLNode[K, V]:
        self._update_height(node)
        bf = self._balance_factor(node)

        # Left heavy
        if bf > 1:
            if self._balance_factor(node.left) < 0:  # LR case
                node.left = self._rotate_left(node.left)  # type: ignore
            return self._rotate_right(node)

        # Right heavy
        if bf < -1:
            if self._balance_factor(node.right) > 0:  # RL case
                node.right = self._rotate_right(node.right)  # type: ignore
            return self._rotate_left(node)

        return node

    def _insert(self, node: Optional[AVLNode[K, V]], key: K, value: V) -> AVLNode[K, V]:
        if node is None:
            return AVLNode(key, value)

        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
        else:
            node.value = value  # update existing key
            return node

        return self._rebalance(node)

    def _find(self, node: Optional[AVLNode[K, V]], key: K) -> Optional[AVLNode[K, V]]:
        while node:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node
        return None

    def _min_value_node(self, node: AVLNode[K, V]) -> AVLNode[K, V]:
        current = node
        while current.left:
            current = current.left
        return current

    def _delete(self, node: Optional[AVLNode[K, V]], key: K) -> Optional[AVLNode[K, V]]:
        if node is None:
            return None

        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            # Node with only one child or no child
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left

            # Node with two children: get inorder successor
            succ = self._min_value_node(node.right)
            node.key, node.value = succ.key, succ.value
            node.right = self._delete(node.right, succ.key)

        return self._rebalance(node) if node else None

    def _inorder(self, node: Optional[AVLNode[K, V]], acc: List[Tuple[K, V]]) -> None:
        if node:
            self._inorder(node.left, acc)
            acc.append((node.key, node.value))
            self._inorder(node.right, acc)

    def _check_balance(self, node: Optional[AVLNode[K, V]]) -> Tuple[int, bool]:
        """Return (height, is_balanced) for subtree rooted at node."""
        if node is None:
            return 0, True
        lh, lb = self._check_balance(node.left)
        rh, rb = self._check_balance(node.right)
        balanced = lb and rb and abs(lh - rh) <= 1
        return 1 + max(lh, rh), balanced
