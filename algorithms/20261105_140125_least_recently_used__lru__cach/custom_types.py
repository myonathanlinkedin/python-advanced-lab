from __future__ import annotations
from typing import Generic, TypeVar, Optional

K = TypeVar('K')
V = TypeVar('V')


class DLinkedNode(Generic[K, V]):
    """
    Doubly linked list node used by LRUCache.
    For sentinel head/tail nodes the key and value are None.
    """
    __slots__ = ('key', 'value', 'prev', 'next')

    def __init__(self, key: Optional[K], value: Optional[V]) -> None:
        self.key: Optional[K] = key
        self.value: Optional[V] = value
        self.prev: Optional['DLinkedNode[K, V]'] = None
        self.next: Optional['DLinkedNode[K, V]'] = None


__all__ = ['DLinkedNode', 'K', 'V']
