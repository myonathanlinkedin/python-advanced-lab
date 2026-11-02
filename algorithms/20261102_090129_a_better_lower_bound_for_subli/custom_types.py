from __future__ import annotations
from typing import List, Tuple, Set, Dict, Iterable, NamedTuple
import math


class Edge(NamedTuple):
    u: int
    v: int
    weight: float


class Graph:
    """
    Simple undirected weighted graph represented by an adjacency matrix.
    The matrix must be symmetric and have zero on the diagonal.
    """
    def __init__(self, dist: List[List[float]]) -> None:
        n = len(dist)
        if any(len(row) != n for row in dist):
            raise ValueError("Distance matrix must be square")
        for i in range(n):
            if not math.isclose(dist[i][i], 0.0, abs_tol=1e-9):
                raise ValueError("Diagonal entries must be zero")
        self.n: int = n
        self.dist: List[List[float]] = dist

    def edges(self) -> Iterable[Edge]:
        """Yield each undirected edge once (u < v)."""
        for i in range(self.n):
            for j in range(i + 1, self.n):
                yield Edge(i, j, self.dist[i][j])

    def metric_closure(self, vertices: Set[int]) -> List[Edge]:
        """Return the complete graph (as edge list) induced by `vertices` using the metric distances."""
        verts = sorted(vertices)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                yield Edge(u, v, self.dist[u][v])
