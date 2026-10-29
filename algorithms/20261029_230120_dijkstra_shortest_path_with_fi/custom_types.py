from __future__ import annotations
from typing import Generic, TypeVar, Optional, List, Tuple, Dict

K = TypeVar('K')
V = TypeVar('V')


class FibHeapNode(Generic[K, V]):
    __slots__ = ('key', 'value', 'degree', 'mark', 'parent', 'child', 'left', 'right')

    def __init__(self, key: K, value: V) -> None:
        self.key: K = key
        self.value: V = value
        self.degree: int = 0
        self.mark: bool = False
        self.parent: Optional[FibHeapNode[K, V]] = None
        self.child: Optional[FibHeapNode[K, V]] = None
        # circular doubly linked list pointers
        self.left: FibHeapNode[K, V] = self
        self.right: FibHeapNode[K, V] = self

    def __repr__(self) -> str:
        return f'FibHeapNode(key={self.key}, value={self.value})'
