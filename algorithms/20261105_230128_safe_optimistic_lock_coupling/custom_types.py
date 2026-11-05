from __future__ import annotations
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional, Generic, TypeVar

T = TypeVar('T')


@dataclass
class Node(Generic[T]):
    """
    A node used by OptimisticLockCouplingLinkedList.
    Each node carries a version counter that is incremented
    under its lock whenever the node's mutable state changes.
    """
    key: T
    next: Optional[Node[T]] = None
    lock: Lock = field(default_factory=Lock, init=False, repr=False)
    version: int = field(default=0, init=False, repr=False)
