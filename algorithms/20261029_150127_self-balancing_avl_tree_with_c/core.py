from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

@dataclass
class Node:
    key: Any
    value: Any
    left: Optional[Node] = None
    right: Optional[Node] = None
    height: int = 1
    timestamp: int = 0

class AVLTree:
    def __init__(self) -> None:
        self.root: Optional[Node] = None
        self._timestamp: int = 0

    def _height(self, node: Optional[Node]) -> int:
        return node.height if node else 0

    def _update(self, node: Node) -> None:
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _balance_factor(self, node: Node) -> int:
        return self._height(node.left) - self._height(node.right)

    def _rotate_right(self, y: Node) -> Node:
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self._update(y)
        self._update(x)
        return x

    def _rotate_left(self, x: Node) -> Node:
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self._update(x)
        self._update(y)
        return y

    def _rebalance(self, node: Node) -> Node:
        self._update(node)
        bf = self._balance_factor(node)
        if bf > 1:
            if self._balance_factor(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if bf < -1:
            if self._balance_factor(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

    def insert(self, key: Any, value: Any) -> None:
        self._timestamp += 1
        self.root = self._insert(self.root, key, value, self._timestamp)

    def _insert(self, node: Optional[Node], key: Any, value: Any, ts: int) -> Node:
        if not node:
            return Node(key, value, timestamp=ts)
        if key < node.key:
            node.left = self._insert(node.left, key, value, ts)
        elif key > node.key:
            node.right = self._insert(node.right, key, value, ts)
        else:
            node.value = value
            node.timestamp = ts
            return node
        return self._rebalance(node)

    def search(self, key: Any) -> Optional[Any]:
        node = self.root
        while node:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node.value
        return None

    def _min_value_node(self, node: Node) -> Node:
        current = node
        while current.left:
            current = current.left
        return current

    def delete(self, key: Any) -> None:
        self.root = self._delete(self.root, key)

    def _delete(self, node: Optional[Node], key: Any) -> Optional[Node]:
        if not node:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if not node.left or not node.right:
                return node.left if node.left else node.right
            temp = self._min_value_node(node.right)
            node.key, node.value, node.timestamp = temp.key, temp.value, temp.timestamp
            node.right = self._delete(node.right, temp.key)
        return self._rebalance(node) if node else None

    def inorder(self) -> List[Tuple[Any, Any, int]]:
        res: List[Tuple[Any, Any, int]] = []
        def _in(node: Optional[Node]) -> None:
            if node:
                _in(node.left)
                res.append((node.key, node.value, node.timestamp))
                _in(node.right)
        _in(self.root)
        return res
