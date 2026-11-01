from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Generic, TypeVar, Optional

T = TypeVar('T')


@dataclass
class BTreeNode(Generic[T]):
    """
    A node of a B-Tree of minimum degree t.
    - keys are kept in sorted order.
    - children length is len(keys)+1 for internal nodes.
    - leaf flag indicates whether the node has children.
    """
    t: int                                            # minimum degree (defines max keys = 2*t-1)
    keys: List[T] = field(default_factory=list)      # stored keys
    children: List[BTreeNode[T]] = field(default_factory=list)  # child pointers
    leaf: bool = True                                 # true if node is a leaf

    def __post_init__(self) -> None:
        # Enforce structural invariants for safety
        max_keys = 2 * self.t - 1
        assert 0 <= len(self.keys) <= max_keys, f"Node keys count {len(self.keys)} exceeds allowed {max_keys}"
        if self.leaf:
            assert not self.children, "Leaf nodes must not have children"
        else:
            assert len(self.children) == len(self.keys) + 1, (
                f"Internal node children count {len(self.children)} must be keys+1 ({len(self.keys)+1})"
            )

    def is_full(self) -> bool:
        """Return True if the node contains the maximum number of keys."""
        return len(self.keys) == 2 * self.t - 1

    def __repr__(self) -> str:
        return (f"BTreeNode(t={self.t}, leaf={self.leaf}, "
                f"keys={self.keys}, children={len(self.children)})")
