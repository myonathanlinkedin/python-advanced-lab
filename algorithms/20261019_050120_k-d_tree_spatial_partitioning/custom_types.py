from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any

# A point in k-dimensional space
Point = Tuple[float, ...]


@dataclass(frozen=True)
class KDNode:
    point: Point
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None
    axis: int = 0  # dimension used for splitting at this node
