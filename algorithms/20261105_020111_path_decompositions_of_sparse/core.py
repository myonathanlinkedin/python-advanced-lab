"""
Core implementation for exact pathwidth computation of undirected graphs.

The algorithm enumerates all vertex subsets (bit‑mask DP) and computes the
minimum possible maximum size of the “active” set during a linear layout.
For a graph G with n vertices the DP runs in O(n·2ⁿ) time and O(2ⁿ) space,
which is optimal for exact exponential algorithms on sparse graphs.

Definitions
-----------
* active set of a prefix S ⊆ V:
    vertices in S that have at least one neighbor in V \ S.
* width of a linear layout = max_{prefix S} |active(S)|.
* pathwidth(G) = min_{linear layouts} (width - 1).

The DP stores for each subset S the smallest achievable width for a layout
that ends exactly with the vertices of S (order of the last vertex is irrelevant
because the width depends only on the set).  The recurrence is:

    DP[∅] = 0
    DP[S] = min_{v∈S} max( DP[S\{v}], |active(S)| )

where |active(S)| can be computed from adjacency bit‑masks.

The returned pathwidth is DP[All] - 1.
"""

from __future__ import annotations
from typing import List, Tuple, Iterable
import itertools


class Graph:
    """Simple undirected graph with vertices indexed 0 … n‑1."""

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("Number of vertices must be non‑negative")
        self.n: int = n
        self._adj_masks: List[int] = [0] * n

    @property
    def adjacency_masks(self) -> List[int]:
        """Return a copy of the adjacency bit‑mask list."""
        return self._adj_masks.copy()

    def add_edge(self, u: int, v: int) -> None:
        """Add an undirected edge (u, v)."""
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise IndexError("Vertex index out of range")
        if u == v:
            # ignore self‑loops – they do not affect pathwidth
            return
        mask_u = 1 << v
        mask_v = 1 << u
        self._adj_masks[u] |= mask_u
        self._adj_masks[v] |= mask_v

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[Tuple[int, int]]) -> "Graph":
        """Factory that builds a graph from an edge list."""
        g = cls(n)
        for u, v in edges:
            g.add_edge(u, v)
        return g

    def neighbors(self, v: int) -> List[int]:
        """Return a list of neighbors of vertex v."""
        mask = self._adj_masks[v]
        return [i for i in range(self.n) if mask & (1 << i)]


def _active_size(mask: int, adj_masks: List[int], all_mask: int) -> int:
    """
    Compute |active(mask)| where `mask` is the set of vertices already placed.
    A vertex is active iff it belongs to `mask` and has a neighbor outside `mask`.
    """
    complement = all_mask ^ mask
    active = 0
    # iterate over bits set in mask
    m = mask
    while m:
        v = (m & -m).bit_length() - 1  # index of least‑significant set bit
        if adj_masks[v] & complement:
            active += 1
        m &= m - 1
    return active


def pathwidth(g: Graph) -> int:
    """
    Exact pathwidth of an undirected graph using O(n·2ⁿ) DP.

    Parameters
    ----------
    g : Graph
        The input graph.

    Returns
    -------
    int
        Pathwidth of `g`.
    """
    n = g.n
    if n == 0:
        return 0
    adj = g.adjacency_masks
    all_mask = (1 << n) - 1
    size = 1 << n
    dp = [n] * size  # initialise with worst possible width
    dp[0] = 0

    # Pre‑compute active size for every subset to avoid recomputation.
    active_cache = [0] * size
    for mask in range(1, size):
        active_cache[mask] = _active_size(mask, adj, all_mask)

    for mask in range(1, size):
        # try removing each possible last vertex v
        sub = mask
        best = n
        while sub:
            v = (sub & -sub).bit_length() - 1
            prev = mask ^ (1 << v)
            width = max(dp[prev], active_cache[mask])
            if width < best:
                best = width
            sub &= sub - 1
        dp[mask] = best

    # Pathwidth = (minimum width) - 1
    return dp[all_mask] - 1


def _brute_force_pathwidth(g: Graph) -> int:
    """
    Exhaustive enumeration of all permutations.
    Used only for verification on tiny graphs (n ≤ 8).
    """
    n = g.n
    if n == 0:
        return 0
    adj = g.adjacency_masks
    all_mask = (1 << n) - 1
    best = n
    for perm in itertools.permutations(range(n)):
        placed = 0
        max_active = 0
        for v in perm:
            placed |= 1 << v
            # recompute active size of current prefix
            complement = all_mask ^ placed
            active = 0
            m = placed
            while m:
                u = (m & -m).bit_length() - 1
                if adj[u] & complement:
                    active += 1
                m &= m - 1
            if active > max_active:
                max_active = active
                if max_active >= best:
                    break  # prune
        if max_active < best:
            best = max_active
    return best - 1


# Exported symbols
__all__ = ["Graph", "pathwidth", "_brute_force_pathwidth"]
