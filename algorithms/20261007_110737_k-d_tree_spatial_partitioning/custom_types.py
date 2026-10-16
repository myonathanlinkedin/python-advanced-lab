from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence

Point = Tuple[float, ...]  # N-dimensional point


@dataclass
class KDNode:
    point: Point
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None
    axis: int = 0

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None
