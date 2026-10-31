from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple, Iterable, List


Vertex = int
Edge = Tuple[Vertex, Vertex]  # always stored as (min, max) for consistency


@dataclass(frozen=True)
class Graph:
    """
    Simple undirected graph represented by an adjacency map.
    The adjacency map stores for each vertex the set of its neighbours.
    """
    adjacency: Dict[Vertex, Set[Vertex]]

    @staticmethod
    def empty() -> Graph:
        return Graph(adjacency={})

    @staticmethod
    def from_edges(vertices: Iterable[Vertex], edges: Iterable[Edge]) -> Graph:
        adj: Dict[Vertex, Set[Vertex]] = {v: set() for v in vertices}
        for u, v in edges:
            if u == v:
                raise ValueError("Self‑loops are not allowed")
            adj[u].add(v)
            adj[v].add(u)
        return Graph(adjacency=adj)

    def vertices(self) -> Set[Vertex]:
        return set(self.adjacency.keys())

    def edges(self) -> Set[Edge]:
        seen: Set[Edge] = set()
        for u, neigh in self.adjacency.items():
            for v in neigh:
                e = (min(u, v), max(u, v))
                seen.add(e)
        return seen

    def degree(self, v: Vertex) -> int:
        return len(self.adjacency.get(v, set()))

    def add_edge(self, u: Vertex, v: Vertex) -> Graph:
        if u == v:
            raise ValueError("Self‑loops are not allowed")
        new_adj = {k: set(vs) for k, vs in self.adjacency.items()}
        for x, y in ((u, v), (v, u)):
            if x not in new_adj:
                new_adj[x] = set()
            new_adj[x].add(y)
        return Graph(adjacency=new_adj)

    def neighbours(self, v: Vertex) -> Set[Vertex]:
        return set(self.adjacency.get(v, set()))
