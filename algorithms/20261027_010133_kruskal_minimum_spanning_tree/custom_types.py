from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, order=True)
class Edge:
    """Immutable representation of an undirected weighted edge."""
    weight: int
    u: int
    v: int

    def __post_init__(self) -> None:
        if self.u == self.v:
            raise ValueError("Self‑loops are not allowed in Edge.")
        if self.weight < 0:
            raise ValueError("Edge weight must be non‑negative.")


@dataclass
class Graph:
    """Simple graph container for Kruskal's algorithm."""
    num_vertices: int
    edges: List[Edge]

    def __post_init__(self) -> None:
        if self.num_vertices < 0:
            raise ValueError("Number of vertices cannot be negative.")
        for e in self.edges:
            if not (0 <= e.u < self.num_vertices) or not (0 <= e.v < self.num_vertices):
                raise ValueError(f"Edge {e} references invalid vertex indices.")
