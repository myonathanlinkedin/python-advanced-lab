from __future__ import annotations

from typing import List, Tuple, Dict, Set, Iterable
from custom_types import Edge, Graph


class DisjointSetUnion:
    """
    Disjoint‑Set Union (Union‑Find) with path compression and union by rank.
    Provides near‑constant amortised time operations.
    """

    def __init__(self, elements: Iterable[str | int]) -> None:
        self.parent: Dict[str | int, str | int] = {}
        self.rank: Dict[str | int, int] = {}
        for e in elements:
            self.parent[e] = e
            self.rank[e] = 0

    def find(self, x: str | int) -> str | int:
        """Find the representative of the set containing x."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: str | int, y: str | int) -> bool:
        """
        Merge the sets containing x and y.
        Returns True if a merge happened, False if they were already in the same set.
        """
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return False

        # Union by rank
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1
        return True

    def same_set(self, x: str | int, y: str | int) -> bool:
        """Check whether x and y belong to the same set."""
        return self.find(x) == self.find(y)


def kruskal_mst(edges: Iterable[Edge]) -> Tuple[List[Edge], float]:
    """
    Compute a Minimum Spanning Forest (MSF) using Kruskal's algorithm.
    The function works for disconnected graphs as well; it returns a forest
    containing a minimum‑weight spanning tree for each connected component.

    Parameters
    ----------
    edges: Iterable[Edge]
        All edges of the undirected graph. Vertices are inferred from the edges.

    Returns
    -------
    mst_edges: List[Edge]
        List of edges belonging to the MSF, sorted by the order they were added.
    total_weight: float
        Sum of the weights of the selected edges.
    """
    # Extract all distinct vertices
    vertices: Set[str | int] = set()
    for e in edges:
        vertices.add(e.u)
        vertices.add(e.v)

    dsu = DisjointSetUnion(vertices)

    # Sort edges deterministically: primary by weight, then by (u, v)
    sorted_edges = sorted(edges)

    mst: List[Edge] = []
    total = 0.0

    for e in sorted_edges:
        if dsu.union(e.u, e.v):
            mst.append(e)
            total += e.weight

    return mst, total
