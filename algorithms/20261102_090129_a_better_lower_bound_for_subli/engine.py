from __future__ import annotations
from typing import List, Set, Tuple
from custom_types import Graph, Edge
import itertools


class UnionFind:
    """Disjoint-set union‑find with path compression and union by rank."""
    def __init__(self, size: int) -> None:
        self.parent: List[int] = list(range(size))
        self.rank: List[int] = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True


def mst_weight(edges: List[Edge], vertex_count: int) -> float:
    """
    Compute the weight of a Minimum Spanning Tree using Kruskal's algorithm.
    `edges` may contain edges not incident to all vertices; the function
    assumes that the graph is connected on the provided vertices.
    """
    uf = UnionFind(vertex_count)
    total = 0.0
    for e in sorted(edges, key=lambda e: e.weight):
        if uf.union(e.u, e.v):
            total += e.weight
    return total


def lower_bound_steiner(graph: Graph, terminals: Set[int]) -> float:
    """
    A simple yet provably valid lower bound for the Metric Steiner Tree problem:
    the weight of the Minimum Spanning Tree on the metric closure of the terminals.
    This bound is tight for many instances and is computable in O(|T|^2 log |T|)
    time where |T| is the number of terminals.
    """
    if len(terminals) <= 1:
        return 0.0
    # Build complete graph on terminals using the metric distances.
    terminal_edges = list(graph.metric_closure(terminals))
    # The vertex ids are the original node numbers; we need a mapping to a dense range.
    mapping = {v: i for i, v in enumerate(sorted(terminals))}
    remapped_edges = [Edge(mapping[e.u], mapping[e.v], e.weight) for e in terminal_edges]
    return mst_weight(remapped_edges, len(terminals))


def exact_steiner_tree_weight(graph: Graph, terminals: Set[int]) -> float:
    """
    Exact Steiner tree weight for small graphs (|V| <= 12) using the classic
    DP over subsets (Dreyfus‑Wagner algorithm). Runs in O(3^k * n + 2^k * n^2)
    where k = |terminals| and n = |V|. Suitable only for verification in unit tests.
    """
    n = graph.n
    k = len(terminals)
    if k == 0:
        return 0.0
    term_list = list(terminals)
    term_index = {v: i for i, v in enumerate(term_list)}

    # DP[mask][v] = min cost of a tree that connects terminals in `mask` and ends at vertex v
    INF = float('inf')
    dp: List[List[float]] = [[INF] * n for _ in range(1 << k)]

    # Base cases: single terminal masks
    for i, t in enumerate(term_list):
        dp[1 << i][t] = 0.0

    # Iterate over all masks
    for mask in range(1, 1 << k):
        # Skip singleton masks (already initialized)
        if mask & (mask - 1) == 0:
            continue
        # Submask iteration for DP merging
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            for v in range(n):
                cand = dp[sub][v] + dp[other][v]
                if cand < dp[mask][v]:
                    dp[mask][v] = cand
            sub = (sub - 1) & mask

        # After merging, run a relaxation over edges (essentially a shortest‑path step)
        # Using Floyd‑Warshall style relaxation because the graph is complete metric.
        for v in range(n):
            for u in range(n):
                if dp[mask][u] + graph.dist[u][v] < dp[mask][v]:
                    dp[mask][v] = dp[mask][u] + graph.dist[u][v]

    full_mask = (1 << k) - 1
    return min(dp[full_mask])
