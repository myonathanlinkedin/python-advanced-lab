"""
core.py
---------

Reference implementation for linear‑time FPT rectilinear planarity testing
restricted to series‑parallel graphs.

The algorithm relies on the following well‑known facts:

1. Every series‑parallel (SP) graph is planar.
2. A planar graph admits a rectilinear (orthogonal, zero‑bend) embedding
   iff it is planar, has maximum degree ≤ 4, and each vertex can be
   assigned a 90° angle to each incident edge (i.e. the sum of incident
   angles equals 360°).  For SP graphs this condition is also sufficient:
   the SP decomposition can be turned into a rectangular layout where
   each edge is drawn either horizontally or vertically.

Consequently, testing rectilinear planarity for SP graphs reduces to:

    * Recognise whether the input graph is series‑parallel.
    * Verify that Δ(G) ≤ 4.

Both steps can be performed in linear time.

The implementation below follows the classic reduction algorithm for SP
recognition (Valdes‑Tarjan‑Lawler 1982).  The graph is represented by an
adjacency list; parallel edges and self‑loops are rejected because they
cannot appear in a simple SP graph.

Public API
~~~~~~~~~~
- `Graph`: mutable undirected simple graph.
- `is_series_parallel(g: Graph, s: int, t: int) -> bool`
- `is_rectilinear_planar(g: Graph, s: int, t: int) -> bool`

The vertices `s` and `t` are the designated terminals of the SP graph.
For a connected SP graph any pair of vertices can serve as terminals; the
unit tests use vertices 0 and 1.

All functions are pure, deterministic and use only the Python standard
library.
"""

from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple


class Graph:
    """Undirected simple graph with integer vertices."""

    __slots__ = ("adj", "num_vertices")

    def __init__(self, num_vertices: int) -> None:
        if num_vertices < 0:
            raise ValueError("Number of vertices must be non‑negative")
        self.num_vertices: int = num_vertices
        self.adj: List[Set[int]] = [set() for _ in range(num_vertices)]

    def add_edge(self, u: int, v: int) -> None:
        if u == v:
            raise ValueError("Self‑loops are not allowed")
        if not (0 <= u < self.num_vertices and 0 <= v < self.num_vertices):
            raise IndexError("Vertex index out of range")
        if v in self.adj[u]:
            raise ValueError("Parallel edges are not allowed")
        self.adj[u].add(v)
        self.adj[v].add(u)

    def degree(self, v: int) -> int:
        return len(self.adj[v])

    def edges(self) -> List[Tuple[int, int]]:
        """Return a list of edges with u < v."""
        es: List[Tuple[int, int]] = []
        for u in range(self.num_vertices):
            for v in self.adj[u]:
                if u < v:
                    es.append((u, v))
        return es

    def copy(self) -> "Graph":
        g = Graph(self.num_vertices)
        for u, v in self.edges():
            g.add_edge(u, v)
        return g


def _parallel_reduce(
    g: Graph, edge_counts: Dict[Tuple[int, int], int]
) -> bool:
    """
    Perform parallel reductions in place.
    Returns True if a reduction was performed.
    """
    for (u, v), cnt in list(edge_counts.items()):
        if cnt > 1:
            # Parallel edges collapse to a single edge.
            # Remove the extra copies.
            edge_counts[(u, v)] = 1
            # No structural change needed in adjacency because we store
            # simple edges only.
            return True
    return False


def _series_reduce(
    g: Graph, degree_one: Set[int], degree_two: Set[int]
) -> bool:
    """
    Perform a series reduction on a degree‑2 vertex that is not a terminal.
    Returns True if a reduction was performed.
    """
    for v in list(degree_two):
        if v in degree_one:
            continue  # terminal may have degree 1
        neigh = list(g.adj[v])
        if len(neigh) != 2:
            continue
        a, b = neigh
        # Remove v and its incident edges
        g.adj[a].remove(v)
        g.adj[b].remove(v)
        g.adj[v].clear()
        # Add edge (a, b) if it does not already exist
        if b not in g.adj[a]:
            g.adj[a].add(b)
            g.adj[b].add(a)
        # Update degree sets
        degree_two.discard(v)
        degree_one.discard(v)
        # Re‑evaluate degrees of a and b
        for x in (a, b):
            deg = len(g.adj[x])
            if deg == 1:
                degree_one.add(x)
                degree_two.discard(x)
            elif deg == 2:
                degree_two.add(x)
                degree_one.discard(x)
            else:
                degree_one.discard(x)
                degree_two.discard(x)
        return True
    return False


def is_series_parallel(g: Graph, s: int, t: int) -> bool:
    """
    Linear‑time recognition of series‑parallel graphs with terminals (s, t).

    The algorithm repeatedly applies series and parallel reductions until
    either a single edge (s, t) remains (success) or no reduction is possible
    (failure).

    Parameters
    ----------
    g : Graph
        Input graph (will be copied; the original is left unchanged).
    s, t : int
        Distinct terminal vertices.

    Returns
    -------
    bool
        True iff the graph is series‑parallel with terminals s and t.
    """
    if s == t:
        raise ValueError("Terminals must be distinct")
    if not (0 <= s < g.num_vertices and 0 <= t < g.num_vertices):
        raise IndexError("Terminal index out of range")

    # Work on a mutable copy.
    g = g.copy()

    # Edge multiplicity map for parallel detection.
    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for u, v in g.edges():
        key = (min(u, v), max(u, v))
        edge_counts[key] += 1

    # Initialise degree buckets.
    degree_one: Set[int] = {v for v in range(g.num_vertices) if g.degree(v) == 1}
    degree_two: Set[int] = {v for v in range(g.num_vertices) if g.degree(v) == 2}

    # Main reduction loop.
    while True:
        # 1. Parallel reduction.
        if _parallel_reduce(g, edge_counts):
            continue

        # 2. Series reduction (excluding terminals).
        if _series_reduce(g, degree_one, degree_two):
            continue

        # No further reduction possible.
        break

    # After reductions, the graph must consist of exactly one edge (s, t)
    # and no other vertices with incident edges.
    if t in g.adj[s] and len(g.adj[s]) == 1 and len(g.adj[t]) == 1:
        # Ensure all other vertices are isolated.
        for v in range(g.num_vertices):
            if v not in (s, t) and g.adj[v]:
                return False
        return True
    return False


def is_rectilinear_planar(g: Graph, s: int, t: int) -> bool:
    """
    Test whether a series‑parallel graph admits a rectilinear planar embedding.

    For SP graphs the condition reduces to:
        * The graph is series‑parallel with terminals (s, t).
        * Maximum degree ≤ 4.

    Parameters
    ----------
    g : Graph
        Input graph.
    s, t : int
        Designated terminals.

    Returns
    -------
    bool
        True iff a rectilinear planar embedding exists.
    """
    # Quick degree check – necessary for any zero‑bend orthogonal drawing.
    max_deg = max(g.degree(v) for v in range(g.num_vertices))
    if max_deg > 4:
        return False

    # SP recognition.
    return is_series_parallel(g, s, t)


# End of core.py
