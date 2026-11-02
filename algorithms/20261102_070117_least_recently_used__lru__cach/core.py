"""
core.py
---------

Implementation of an LRU (Least Recently Used) cache using a hash map
combined with a doubly linked list to achieve O(1) get and put operations.

The cache stores key-value pairs where keys are hashable and values can be
any Python object. When the cache exceeds its capacity, the least recently
used entry is evicted.

Public Interface
----------------
- LRUCache(capacity: int)
    * get(key) -> Any | None
    * put(key, value) -> None
    * __len__() -> int
    * __contains__(key) -> bool
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class _Node:
    """Internal doubly linked list node."""
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Any, value: Any) -> None:
        self.key: Any = key
        self.value: Any = value
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class LRUCache:
    """
    LRU Cache with O(1) get and put.

    Attributes
    ----------
    capacity : int
        Maximum number of items the cache can hold. Must be >= 0.
    """

    __slots__ = ("_capacity", "_map", "_head", "_tail", "_size")

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non‑negative")
        self._capacity: int = capacity
        self._map: Dict[Any, _Node] = {}
        # Dummy head and tail simplify edge handling.
        self._head: _Node = _Node(None, None)  # type: ignore
        self._tail: _Node = _Node(None, None)  # type: ignore
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size: int = 0

    # --------------------------------------------------------------------- #
    # Internal linked‑list helpers
    # --------------------------------------------------------------------- #
    def _add_node_to_front(self, node: _Node) -> None:
        """Insert node right after head (most‑recent position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node  # type: ignore
        self._head.next = node
        self._size += 1

    def _remove_node(self, node: _Node) -> None:
        """Detach node from the list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node is None or next_node is None:
            # Should never happen for real nodes.
            raise RuntimeError("Attempted to remove a detached node")
        prev_node.next = next_node
        next_node.prev = prev_node
        node.prev = node.next = None
        self._size -= 1

    def _move_to_front(self, node: _Node) -> None:
        """Move an existing node to the most‑recent position."""
        self._remove_node(node)
        self._add_node_to_front(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the least‑recently used node."""
        lru = self._tail.prev
        if lru is self._head:
            raise RuntimeError("Attempted to pop from an empty cache")
        self._remove_node(lru)  # type: ignore
        return lru  # type: ignore

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get(self, key: Any) -> Any | None:
        """
        Retrieve the value associated with *key*.

        Returns
        -------
        The cached value if present; otherwise ``None``.
        """
        node = self._map.get(key)
        if node is None:
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any) -> None:
        """
        Insert or update the *key* with *value*.

        If the cache exceeds its capacity, the least recently used entry
        is evicted.
        """
        node = self._map.get(key)
        if node:
            node.value = value
            self._move_to_front(node)
            return

        if self._capacity == 0:
            # Cache cannot store items; silently ignore.
            return

        new_node = _Node(key, value)
        self._map[key] = new_node
        self._add_node_to_front(new_node)

        if self._size > self._capacity:
            lru = self._pop_tail()
            del self._map[lru.key]

    def __len__(self) -> int:
        """Current number of items stored."""
        return self._size

    def __contains__(self, key: Any) -> bool:
        """Return ``True`` if *key* is present in the cache."""
        return key in self._map

    def __repr__(self) -> str:
        items = []
        cur = self._head.next
        while cur is not self._tail:
            items.append(f"{cur.key!r}: {cur.value!r}")
            cur = cur.next  # type: ignore
        return f"LRUCache(capacity={self._capacity}, items={{" + ", ".join(items) + "}})"
