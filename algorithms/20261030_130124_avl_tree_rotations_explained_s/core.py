from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Generator

@dataclass
class Node:
    key: int
    left: Optional[Node] = None
    right: Optional[Node] = None
    height: int = 1

class AVLTree:
    def __init__(self) -> None:
        self.root: Optional[Node] = None

    # Public API
    def insert(self, key: int) -> None:
        self.root = self._insert(self.root, key)

    def delete(self, key: int) -> None:
        self.root = self._delete(self.root, key)

    def __contains__(self, key: int) -> bool:
        return self._search(self.root, key) is not None

    def __iter__(self) -> Generator[int, None, None]:
        yield from self._in_order(self.root)

    def __len__(self) -> int:
        return self._size(self.root)

    # Internal helpers
    def _height(self, node: Optional[Node]) -> int:
        return node.height if node else 0

    def _update_height(self, node: Node) -> None:
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _balance_factor(self, node: Node) -> int:
        return self._height(node.left) - self._height(node.right)

    def _rotate_left(self, z: Node) -> Node:
        y = z.right
        if y is None:
            return z
        T2 = y.left
        y.left = z
        z.right = T2
        self._update_height(z)
        self._update_height(y)
        return y

    def _rotate_right(self, z: Node) -> Node:
        y = z.left
        if y is None:
            return z
        T3 = y.right
        y.right = z
        z.left = T3
        self._update_height(z)
        self._update_height(y)
        return y

    def _rebalance(self, node: Node) -> Node:
        self._update_height(node)
        balance = self._balance_factor(node)

        # Left heavy
        if balance > 1:
            if self._balance_factor(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        # Right heavy
        if balance < -1:
            if self._balance_factor(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def _insert(self, node: Optional[Node], key: int) -> Node:
        if node is None:
            return Node(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:
            node.right = self._insert(node.right, key)
        else:
            return node  # Duplicate keys not inserted
        return self._rebalance(node)

    def _min_value_node(self, node: Node) -> Node:
        current = node
        while current.left:
            current = current.left
        return current

    def _delete(self, node: Optional[Node], key: int) -> Optional[Node]:
        if node is None:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None or node.right is None:
                temp = node.left if node.left else node.right
                if temp is None:
                    return None
                node = temp
            else:
                temp = self._min_value_node(node.right)
                node.key = temp.key
                node.right = self._delete(node.right, temp.key)
        return self._rebalance(node) if node else None

    def _search(self, node: Optional[Node], key: int) -> Optional[Node]:
        if node is None or node.key == key:
            return node
        return self._search(node.left, key) if key < node.key else self._search(node.right, key)

    def _in_order(self, node: Optional[Node]) -> Generator[int, None, None]:
        if node:
            yield from self._in_order(node.left)
            yield node.key
            yield from self._in_order(node.right)

    def _size(self, node: Optional[Node]) -> int:
        if node is None:
            return 0
        return 1 + self._size(node.left) + self._size(node.right)

    def __repr__(self) -> str:
        keys = list(self)
        return f"AVLTree({keys})"
