from __future__ import annotations
from enum import Enum
from typing import Any, Optional, Generator, Tuple, List


class Color(Enum):
    RED = 0
    BLACK = 1


class RBNode:
    __slots__ = ("key", "value", "color", "left", "right", "parent")

    def __init__(
        self,
        key: Any,
        value: Any,
        color: Color = Color.RED,
        left: Optional[RBNode] = None,
        right: Optional[RBNode] = None,
        parent: Optional[RBNode] = None,
    ) -> None:
        self.key = key
        self.value = value
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent


class RedBlackTree:
    """
    Deterministic Red‑Black Tree implementation with internal balance assertions.
    Supports insert, delete, search and inorder traversal.
    """

    def __init__(self) -> None:
        self.root: Optional[RBNode] = None
        self._size = 0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        return self.search(key) is not None

    def insert(self, key: Any, value: Any) -> None:
        """Insert a key/value pair. Raises AssertionError if tree invariants break."""
        node = RBNode(key, value)
        y: Optional[RBNode] = None
        x = self.root

        while x is not None:
            y = x
            if node.key < x.key:
                x = x.left
            elif node.key > x.key:
                x = x.right
            else:
                raise ValueError(f"Duplicate key insertion attempted: {key}")

        node.parent = y
        if y is None:
            self.root = node
        elif node.key < y.key:
            y.left = node
        else:
            y.right = node

        node.left = None
        node.right = None
        node.color = Color.RED

        self._insert_fixup(node)
        self._size += 1
        self._assert_properties()

    def delete(self, key: Any) -> None:
        """Delete node with given key. Raises KeyError if key not present."""
        z = self._find_node(key)
        if z is None:
            raise KeyError(key)

        y = z
        y_original_color = y.color
        if z.left is None:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is None:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                if x:
                    x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                if y.right:
                    y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            if y.left:
                y.left.parent = y
            y.color = z.color

        if y_original_color == Color.BLACK:
            self._delete_fixup(x)
        self._size -= 1
        self._assert_properties()

    def search(self, key: Any) -> Optional[Any]:
        node = self._find_node(key)
        return None if node is None else node.value

    def inorder(self) -> List[Tuple[Any, Any]]:
        """Return list of (key, value) pairs in sorted order."""
        result: List[Tuple[Any, Any]] = []
        self._inorder_walk(self.root, result)
        return result

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _find_node(self, key: Any) -> Optional[RBNode]:
        x = self.root
        while x is not None:
            if key < x.key:
                x = x.left
            elif key > x.key:
                x = x.right
            else:
                return x
        return None

    def _inorder_walk(self, node: Optional[RBNode], out: List[Tuple[Any, Any]]) -> None:
        if node is not None:
            self._inorder_walk(node.left, out)
            out.append((node.key, node.value))
            self._inorder_walk(node.right, out)

    # Rotations ----------------------------------------------------------- #
    def _left_rotate(self, x: RBNode) -> None:
        y = x.right
        if y is None:
            return
        x.right = y.left
        if y.left:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, y: RBNode) -> None:
        x = y.left
        if x is None:
            return
        y.left = x.right
        if x.right:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y is y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x

    # Insert fix‑up ------------------------------------------------------- #
    def _insert_fixup(self, z: RBNode) -> None:
        while z.parent and z.parent.color == Color.RED:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right
                if y and y.color == Color.RED:
                    z.parent.color = Color.BLACK
                    y.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    self._right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y and y.color == Color.RED:
                    z.parent.color = Color.BLACK
                    y.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    self._left_rotate(z.parent.parent)
        self.root.color = Color.BLACK

    # Delete helpers ------------------------------------------------------ #
    def _transplant(self, u: RBNode, v: Optional[RBNode]) -> None:
        if u.parent is None:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        if v:
            v.parent = u.parent

    def _minimum(self, node: RBNode) -> RBNode:
        while node.left is not None:
            node = node.left
        return node

    def _delete_fixup(self, x: Optional[RBNode]) -> None:
        while x != self.root and (x is None or x.color == Color.BLACK):
            if x == x.parent.left:
                w = x.parent.right
                if w and w.color == Color.RED:
                    w.color = Color.BLACK
                    x.parent.color = Color.RED
                    self._left_rotate(x.parent)
                    w = x.parent.right
                if (w.left is None or w.left.color == Color.BLACK) and (
                    w.right is None or w.right.color == Color.BLACK
                ):
                    w.color = Color.RED
                    x = x.parent
                else:
                    if w.right is None or w.right.color == Color.BLACK:
                        if w.left:
                            w.left.color = Color.BLACK
                        w.color = Color.RED
                        self._right_rotate(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = Color.BLACK
                    if w.right:
                        w.right.color = Color.BLACK
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if w and w.color == Color.RED:
                    w.color = Color.BLACK
                    x.parent.color = Color.RED
                    self._right_rotate(x.parent)
                    w = x.parent.left
                if (w.right is None or w.right.color == Color.BLACK) and (
                    w.left is None or w.left.color == Color.BLACK
                ):
                    w.color = Color.RED
                    x = x.parent
                else:
                    if w.left is None or w.left.color == Color.BLACK:
                        if w.right:
                            w.right.color = Color.BLACK
                        w.color = Color.RED
                        self._left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = Color.BLACK
                    if w.left:
                        w.left.color = Color.BLACK
                    self._right_rotate(x.parent)
                    x = self.root
        if x:
            x.color = Color.BLACK

    # --------------------------------------------------------------------- #
    # Verification utilities
    # --------------------------------------------------------------------- #
    def _assert_properties(self) -> None:
        """Assert all red‑black invariants; raises AssertionError on violation."""
        if self.root:
            assert self.root.color == Color.BLACK, "Root must be black"

        def dfs(node: Optional[RBNode]) -> int:
            if node is None:
                return 1  # Nil leaves are black, contribute 1 to black-height
            if node.color == Color.RED:
                assert (
                    (node.left is None or node.left.color == Color.BLACK)
                    and (node.right is None or node.right.color == Color.BLACK)
                ), "Red node cannot have red child"
            left_black = dfs(node.left)
            right_black = dfs(node.right)
            assert left_black == right_black, (
                f"Black-height mismatch at key {node.key}: "
                f"left {left_black}, right {right_black}"
            )
            return left_black + (1 if node.color == Color.BLACK else 0)

        dfs(self.root)
