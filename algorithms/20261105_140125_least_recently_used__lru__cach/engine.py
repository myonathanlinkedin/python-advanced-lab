from typing import Generic, TypeVar, Dict, Optional
from custom_types import DLinkedNode

K = TypeVar('K')
V = TypeVar('V')


class LRUCache(Generic[K, V]):
    """
    LRU Cache implementation with O(1) get and put operations.
    Internally uses a hash map for fast key lookup and a doubly linked list
    to track usage order (most‑recently used at the head, least‑recently used
    just before the tail sentinel).
    """
    __slots__ = ('capacity', 'cache', 'head', 'tail')

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non‑negative")
        self.capacity: int = capacity
        self.cache: Dict[K, DLinkedNode[K, V]] = {}

        # Sentinel nodes simplify add/remove logic.
        self.head: DLinkedNode[K, V] = DLinkedNode(None, None)
        self.tail: DLinkedNode[K, V] = DLinkedNode(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    # -----------------------------------------------------------------
    # Internal helper methods manipulating the doubly linked list.
    # -----------------------------------------------------------------
    def _add_node(self, node: DLinkedNode[K, V]) -> None:
        """Insert node right after the head sentinel (most‑recent position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node  # type: ignore[arg-type]
        self.head.next = node

    def _remove_node(self, node: DLinkedNode[K, V]) -> None:
        """Detach node from the list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node is not None:
            prev_node.next = next_node
        if next_node is not None:
            next_node.prev = prev_node
        node.prev = node.next = None

    def _move_to_head(self, node: DLinkedNode[K, V]) -> None:
        """Mark node as most recently used."""
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> DLinkedNode[K, V]:
        """Remove and return the least recently used node."""
        lru_node = self.tail.prev
        if lru_node is self.head or lru_node is None:
            raise RuntimeError("Attempted to pop from an empty cache")
        self._remove_node(lru_node)
        return lru_node

    # -----------------------------------------------------------------
    # Public API.
    # -----------------------------------------------------------------
    def get(self, key: K) -> Optional[V]:
        """
        Retrieve value associated with *key*.
        Returns None if key is absent.
        """
        node = self.cache.get(key)
        if node is None:
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: K, value: V) -> None:
        """
        Insert or update the value for *key*.
        If the cache exceeds its capacity, the least recently used entry
        is evicted.
        """
        if self.capacity == 0:
            return

        node = self.cache.get(key)
        if node:
            node.value = value
            self._move_to_head(node)
        else:
            new_node = DLinkedNode(key, value)
            self.cache[key] = new_node
            self._add_node(new_node)

            if len(self.cache) > self.capacity:
                lru_node = self._pop_tail()
                # lru_node.key is guaranteed to be not None here.
                del self.cache[lru_node.key]  # type: ignore[arg-type]


__all__ = ['LRUCache']
