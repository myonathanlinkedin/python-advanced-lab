from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

K = TypeVar('K')
V = TypeVar('V')


@dataclass
class Node(Generic[K, V]):
    """Doubly linked list node used by LRUCache."""
    key: K
    value: V
    prev: Optional[Node[K, V]] = field(default=None, repr=False)
    next: Optional[Node[K, V]] = field(default=None, repr=False)
