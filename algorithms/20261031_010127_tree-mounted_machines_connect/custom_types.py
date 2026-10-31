from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class TreeNode:
    """
    Represents a single machine mounted on a tree.
    Each node tracks the number of times it has been "knocked".
    """
    node_id: str
    parent_id: Optional[str] = None
    children: Set[str] = field(default_factory=set)
    knock_count: int = 0

    def knock(self) -> None:
        """Register a single knock on this node."""
        self.knock_count += 1

    def reset(self) -> None:
        """Reset the knock counter to zero."""
        self.knock_count = 0
