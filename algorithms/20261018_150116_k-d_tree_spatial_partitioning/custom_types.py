from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

# A point in k-dimensional space is represented as a tuple of floats.
Point = Tuple[float, ...]


@dataclass
class KDNode:
    """
    Node of a k-d tree.

    Attributes
    ----------
    point : Point
        The k-dimensional point stored at this node.
    left : Optional[KDNode]
        Subtree containing points with coordinate[axis] < point[axis].
    right : Optional[KDNode]
        Subtree containing points with coordinate[axis] >= point[axis].
    axis : int
        Axis used for splitting at this node (0 <= axis < k).
    """
    point: Point
    left: Optional[KDNode] = None
    right: Optional[KDNode] = None
    axis: int = 0
