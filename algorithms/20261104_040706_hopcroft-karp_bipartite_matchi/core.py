from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

INF = 10 ** 9


def hopcroft_karp(graph: Dict[int, List[int]]) -> Dict[int, Optional[int]]:
    """
    Computes a maximum cardinality matching in a bipartite graph.

    Parameters
    ----------
    graph : Dict[int, List[int]]
        Adjacency list representation of the bipartite graph.
        Keys are left‑hand side vertices; values are lists of right‑hand side vertices.

    Returns
    -------
    Dict[int, Optional[int]]
        Mapping from each left vertex to its matched right vertex.
        Unmatched left vertices map to None.
    """
    # Ensure adjacency lists are sets to avoid duplicate edges
    adj: Dict[int, Set[int]] = {u: set(vs) for u, vs in graph.items()}

    pair_u: Dict[int, Optional[int]] = {u: None for u in adj}
    pair_v: Dict[int, Optional[int]] = {}
    dist: Dict[int, int] = {}

    # Collect all right vertices
    for vs in adj.values():
        for v in vs:
            if v not in pair_v:
                pair_v[v] = None

    def bfs() -> bool:
        from collections import deque

        queue = deque()
        for u in adj:
            if pair_u[u] is None:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found_free = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu is None:
                    found_free = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found_free

    def dfs(u: int) -> bool:
        for v in adj[u]:
            pu = pair_v[v]
            if pu is None or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching_size = 0
    while bfs():
        for u in adj:
            if pair_u[u] is None and dfs(u):
                matching_size += 1

    return pair_u


def matching_size(graph: Dict[int, List[int]]) -> int:
    """
    Returns the size of a maximum matching in the bipartite graph.
    """
    return sum(1 for v in hopcroft_karp(graph).values() if v is not None)
