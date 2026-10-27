from __future__ import annotations
from typing import List, Tuple

from custom_types import Edge, Graph


class DisjointSetUnion:
    """Union‑Find structure with path compression and union by rank."""

    def __init__(self, size: int) -> None:
        self.parent: List[int] = list(range(size))
        self.rank: List[int] = [0] * size

    def find(self, x: int) -> int:
        """Find the representative of the set containing x."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union the sets of x and y. Return True iff a merge happened."""
        xr = self.find(x)
        yr = self.find(y)
        if xr == yr:
            return False  # Already in the same set
        # Union by rank
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True


def kruskal_mst(graph: Graph) -> Tuple[List[Edge], int]:
    """
    Compute a Minimum Spanning Forest of an undirected weighted graph using Kruskal's algorithm.

    Returns:
        A tuple (mst_edges, total_weight) where mst_edges is a list of edges belonging to the
        minimum spanning forest and total_weight is the sum of their weights.
    """
    if graph.num_vertices == 0:
        return [], 0

    # Sort edges by non‑decreasing weight
    sorted_edges = sorted(graph.edges, key=lambda e: e.weight)

    dsu = DisjointSetUnion(graph.num_vertices)
    mst_edges: List[Edge] = []
    total_weight = 0

    for edge in sorted_edges:
        if dsu.union(edge.u, edge.v):
            mst_edges.append(edge)
            total_weight += edge.weight
            # Early exit: a forest with (V - C) edges where C is number of components.
            # When the forest becomes spanning for each component, no more unions are possible.
            if len(mst_edges) == graph.num_vertices - 1:
                # In a connected graph this is the full MST; in a disconnected graph we may stop earlier.
                # The condition above is safe because further unions would be impossible.
                break

    return mst_edges, total_weight
