from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict, Set


@dataclass(frozen=True, order=True)
class Edge:
    """
    Represents an undirected weighted edge between two vertices.
    The vertices are identified by hashable objects (commonly int or str).
    The dataclass is ordered by weight first, then by the tuple (u, v)
    to guarantee deterministic sorting when weights are equal.
    """
    weight: float
    u: str | int
    v: str | int

    def __post_init__(self) -> None:
        if self.u == self.v:
            raise ValueError("Self‑loops are not allowed in Edge definitions.")
        # Ensure a canonical orientation for deterministic equality/comparison
        object.__setattr__(self, "u", min(self.u, self.v))
        object.__setattr__(self, "v", max(self.u, self.v))


Graph = Dict[str, Set[Tuple[str, float]]]  # adjacency list: vertex -> set of (neighbor, weight)
