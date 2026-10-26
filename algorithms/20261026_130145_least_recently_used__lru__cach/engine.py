from __future__ import annotations

from typing import Dict, Generic, Optional, TypeVar

from custom_types import Node, K, V


class LRUCache(Generic[K, V]):
    """
    Least Recently Used (LRU) cache implementation with O(1) get and put operations.
    Internally uses a hash map for fast key lookup and a doubly linked list to track usage order.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("Capacity must be non‑negative")
        self.capacity: int = capacity
        self.cache: Dict[K, Node[K, V]] = {}
        # Dummy head and tail simplify edge handling.
        self.head: Node[K, V] = Node(None, None)  # type: ignore
        self.tail: Node[K, V] = Node(None, None)  # type: ignore
        self.head.next = self.tail
        self.tail.prev = self.head

    # --------------------------------------------------------------------- #
    # Internal helper methods for list manipulation
    # --------------------------------------------------------------------- #
    def _add_node(self, node: Node[K, V]) -> None:
        """Insert node right after head (most recently used position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node  # type: ignore
        self.head.next = node

    def _remove_node(self, node: Node[K, V]) -> None:
        """Detach node from the linked list."""
        prev = node.prev
        nxt = node.next
        if prev is not None:
            prev.next = nxt
        if nxt is not None:
            nxt.prev = prev
        node.prev = node.next = None

    def _move_to_head(self, node: Node[K, V]) -> None:
        """Mark node as recently used."""
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> Node[K, V]:
        """Remove and return the least recently used node."""
        lru = self.tail.prev
        if lru is self.head or lru is None:
            raise RuntimeError("Attempted to pop from an empty LRU cache")
        self._remove_node(lru)
        return lru

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get(self, key: K) -> Optional[V]:
        """
        Retrieve value associated with *key*.
        Returns None if key is absent (mirrors typical LRU cache semantics where -1 is used;
        the caller may translate None to -1 if desired).
        """
        node = self.cache.get(key)
        if not node:
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: K, value: V) -> None:
        """
        Insert or update the value for *key*.
        If the cache exceeds its capacity, the least recently used entry is evicted.
        """
        if self.capacity == 0:
            return  # No storage possible.

        node = self.cache.get(key)
        if node:
            node.value = value
            self._move_to_head(node)
        else:
            new_node = Node(key, value)
            self.cache[key] = new_node
            self._add_node(new_node)

            if len(self.cache) > self.capacity:
                lru = self._pop_tail()
                del self.cache[lru.key]

    def __len__(self) -> int:
        """Current number of stored items."""
        return len(self.cache)

    def __contains__(self, key: K) -> bool:
        return key in self.cache

    def keys(self) -> list[K]:
        """Return keys ordered from most to least recently used."""
        result: list[K] = []
        cur = self.head.next
        while cur and cur is not self.tail:
            result.append(cur.key)
            cur = cur.next
        return result
