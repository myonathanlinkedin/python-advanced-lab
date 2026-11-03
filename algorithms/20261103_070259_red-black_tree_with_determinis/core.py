from __future__ import annotations
from typing import Any, Optional, List, Tuple

RED = True
BLACK = False


class Node:
    __slots__ = ("key", "value", "color", "left", "right", "parent")

    def __init__(
        self,
        key: Any = None,
        value: Any = None,
        color: bool = BLACK,
        left: Optional[Node] = None,
        right: Optional[Node] = None,
        parent: Optional[Node] = None,
    ) -> None:
        self.key = key
        self.value = value
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent

    def __repr__(self) -> str:
        return f"Node({self.key!r}, {self.color and 'R' or 'B'})"


# Sentinel NIL node shared by all leaves
NIL = Node()
NIL.left = NIL.right = NIL.parent = NIL
NIL.color = BLACK


class RBTree:
    """Red‑Black Tree with deterministic balance assertions."""

    __slots__ = ("root",)

    def __init__(self) -> None:
        self.root: Node = NIL

    # ---------- Public API ----------
    def insert(self, key: Any, value: Any = None) -> None:
        """Insert key/value pair, overwriting existing key."""
        node = Node(key, value, RED, NIL, NIL, NIL)
        y = NIL
        x = self.root
        while x is not NIL:
            y = x
            if node.key < x.key:
                x = x.left
            elif node.key > x.key:
                x = x.right
            else:
                x.value = value
                return
        node.parent = y
        if y is NIL:
            self.root = node
        elif node.key < y.key:
            y.left = node
        else:
            y.right = node
        self._fix_insert(node)
        self._assert_properties()

    def delete(self, key: Any) -> None:
        """Delete node with given key; raises KeyError if absent."""
        z = self._search_node(key)
        if z is NIL:
            raise KeyError(key)
        y = z
        y_original_color = y.color
        if z.left is NIL:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is NIL:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        if y_original_color is BLACK:
            self._fix_delete(x)
        self._assert_properties()

    def get(self, key: Any) -> Any:
        """Return value associated with key; raises KeyError if absent."""
        node = self._search_node(key)
        if node is NIL:
            raise KeyError(key)
        return node.value

    def inorder(self) -> List[Tuple[Any, Any]]:
        """Return list of (key, value) pairs in sorted order."""
        result: List[Tuple[Any, Any]] = []

        def _inorder(node: Node) -> None:
            if node is NIL:
                return
            _inorder(node.left)
            result.append((node.key, node.value))
            _inorder(node.right)

        _inorder(self.root)
        return result

    # ---------- Internal helpers ----------
    def _search_node(self, key: Any) -> Node:
        x = self.root
        while x is not NIL and key != x.key:
            x = x.left if key < x.key else x.right
        return x

    def _minimum(self, node: Node) -> Node:
        while node.left is not NIL:
            node = node.left
        return node

    def _transplant(self, u: Node, v: Node) -> None:
        if u.parent is NIL:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _rotate_left(self, x: Node) -> None:
        y = x.right
        x.right = y.left
        if y.left is not NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is NIL:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _rotate_right(self, y: Node) -> None:
        x = y.left
        y.left = x.right
        if x.right is not NIL:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is NIL:
            self.root = x
        elif y is y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x

    def _fix_insert(self, z: Node) -> None:
        while z.parent.color is RED:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right
                if y.color is RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._rotate_left(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_right(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color is RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._rotate_right(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_left(z.parent.parent)
        self.root.color = BLACK

    def _fix_delete(self, x: Node) -> None:
        while x is not self.root and x.color is BLACK:
            if x is x.parent.left:
                w = x.parent.right
                if w.color is RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_left(x.parent)
                    w = x.parent.right
                if w.left.color is BLACK and w.right.color is BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color is BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self._rotate_right(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._rotate_left(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if w.color is RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_right(x.parent)
                    w = x.parent.left
                if w.right.color is BLACK and w.left.color is BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color is BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self._rotate_left(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._rotate_right(x.parent)
                    x = self.root
        x.color = BLACK

    # ---------- Assertions ----------
    def _assert_properties(self) -> None:
        """Validate red‑black invariants; raises AssertionError on violation."""
        assert self.root is NIL or self.root.color is BLACK, "Root must be black"
        def _check(node: Node) -> Tuple[int, bool]:
            if node is NIL:
                return 1, True  # black-height of leaf is 1 (the NIL itself)
            left_bh, left_ok = _check(node.left)
            right_bh, right_ok = _check(node.right)
            assert left_ok and right_ok, "Subtree violation"
            assert left_bh == right_bh, f"Black-height mismatch at key {node.key}"
            if node.color is RED:
                assert node.left.color is BLACK and node.right.color is BLACK, \
                    f"Red node {node.key} has red child"
            bh = left_bh + (1 if node.color is BLACK else 0)
            return bh, True
        _check(self.root)
