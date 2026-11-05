"""
core.py
---------

Pure‑Python implementation of an undirected/directed graph with
Breadth‑First Search (BFS) and Depth‑First Search (DFS) traversals.

The module is deliberately minimal, uses only the Python standard library,
and is fully type‑annotated for static analysis.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Generic, Iterable, List, Set, TypeVar, Optional

T = TypeVar("T", bound=hashable)


class Graph(Generic[T]):
    """
    Simple adjacency‑list graph.

    Parameters
    ----------
    directed: bool, optional
        If True the graph is directed; otherwise edges are added in both
        directions.
    """

    def __init__(self, directed: bool = False) -> None:
        self._adj: Dict[T, Set[T]] = {}
        self.directed = directed

    # --------------------------------------------------------------------- #
    # Mutating API
    # --------------------------------------------------------------------- #
    def add_node(self, node: T) -> None:
        """Insert a node if it does not already exist."""
        self._adj.setdefault(node, set())

    def add_edge(self, u: T, v: T) -> None:
        """
        Add an edge (u, v).  For undirected graphs the reverse edge is added
        automatically.
        """
        self.add_node(u)
        self.add_node(v)
        self._adj[u].add(v)
        if not self.directed:
            self._adj[v].add(u)

    # --------------------------------------------------------------------- #
    # Query API
    # --------------------------------------------------------------------- #
    def neighbors(self, node: T) -> Set[T]:
        """Return the set of adjacent vertices of *node*."""
        if node not in self._adj:
            raise KeyError(f"Node {node!r} is not present in the graph.")
        return self._adj[node]

    def nodes(self) -> Set[T]:
        """Return all nodes of the graph."""
        return set(self._adj.keys())

    # --------------------------------------------------------------------- #
    # Traversal algorithms
    # --------------------------------------------------------------------- #
    def bfs(self, start: T) -> List[T]:
        """
        Breadth‑First Search.

        Returns a list of vertices in the order they are first visited.
        Raises ValueError if *start* is not a member of the graph.
        """
        if start not in self._adj:
            raise ValueError(f"Start vertex {start!r} not in graph.")
        visited: Set[T] = set()
        order: List[T] = []
        q: deque[T] = deque([start])
        visited.add(start)

        while q:
            node = q.popleft()
            order.append(node)
            for nbr in self._adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    q.append(nbr)
        return order

    def dfs(self, start: T) -> List[T]:
        """
        Depth‑First Search (recursive).

        Returns a list of vertices in the order they are first visited.
        Raises ValueError if *start* is not a member of the graph.
        """
        if start not in self._adj:
            raise ValueError(f"Start vertex {start!r} not in graph.")
        visited: Set[T] = set()
        order: List[T] = []

        def _visit(v: T) -> None:
            visited.add(v)
            order.append(v)
            for nbr in self._adj[v]:
                if nbr not in visited:
                    _visit(nbr)

        _visit(start)
        return order

    def dfs_iter(self, start: T) -> List[T]:
        """
        Depth‑First Search (iterative, using an explicit stack).

        Returns a list of vertices in the order they are first visited.
        Raises ValueError if *start* is not a member of the graph.
        """
        if start not in self._adj:
            raise ValueError(f"Start vertex {start!r} not in graph.")
        visited: Set[T] = set()
        order: List[T] = []
        stack: List[T] = [start]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            # Push neighbors in reverse sorted order for deterministic output
            for nbr in sorted(self._adj[node], reverse=True):
                if nbr not in visited:
                    stack.append(nbr)
        return order

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #
    def path_exists(self, src: T, dst: T, method: str = "bfs") -> bool:
        """
        Determine whether a path exists between *src* and *dst* using the
        specified traversal method ('bfs' or 'dfs').

        Returns True if reachable, False otherwise.
        """
        if src not in self._adj or dst not in self._adj:
            return False
        traversal = self.bfs if method == "bfs" else self.dfs
        return dst in traversal(src)


# ------------------------------------------------------------------------- #
# Public helper functions (module level)
# ------------------------------------------------------------------------- #
def bfs(graph: Graph[T], start: T) -> List[T]:
    """Convenient functional wrapper around Graph.bfs."""
    return graph.bfs(start)


def dfs(graph: Graph[T], start: T) -> List[T]:
    """Convenient functional wrapper around Graph.dfs."""
    return graph.dfs(start)


def dfs_iter(graph: Graph[T], start: T) -> List[T]:
    """Convenient functional wrapper around Graph.dfs_iter."""
    return graph.dfs_iter(start)


# End of core.py
