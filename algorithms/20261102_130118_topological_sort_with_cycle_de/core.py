"""
core.py
---------

Provides a minimal, production‑grade implementation of a directed graph with
topological sorting and cycle detection.

Public API
~~~~~~~~~~
- DirectedGraph: mutable directed graph class.
- CycleError: exception raised when a cycle is detected during topological sort.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Set, List, Hashable, Iterable


class CycleError(Exception):
    """Raised when a cycle is found in a directed graph during topological sort."""
    pass


class DirectedGraph:
    """
    Simple directed graph using adjacency sets.

    Nodes are hashable objects. The graph allows isolated nodes (no edges).
    """

    __slots__ = ("_adj",)

    def __init__(self, edges: Iterable[tuple[Hashable, Hashable]] | None = None) -> None:
        """
        Initialise the graph.

        Parameters
        ----------
        edges: optional iterable of (source, target) pairs.
        """
        self._adj: Dict[Hashable, Set[Hashable]] = {}
        if edges:
            for src, dst in edges:
                self.add_edge(src, dst)

    def add_node(self, node: Hashable) -> None:
        """Ensure *node* exists in the graph."""
        self._adj.setdefault(node, set())

    def add_edge(self, src: Hashable, dst: Hashable) -> None:
        """Add a directed edge src → dst, creating nodes as needed."""
        self.add_node(src)
        self.add_node(dst)
        self._adj[src].add(dst)

    def nodes(self) -> Set[Hashable]:
        """Return a set of all nodes in the graph."""
        return set(self._adj.keys())

    def edges(self) -> Set[tuple[Hashable, Hashable]]:
        """Return a set of all directed edges."""
        return {(src, dst) for src, targets in self._adj.items() for dst in targets}

    def topological_sort(self) -> List[Hashable]:
        """
        Return a topological ordering of the graph's nodes.

        Raises
        ------
        CycleError
            If the graph contains at least one directed cycle.
        """
        # Compute indegree for each node.
        indegree: Dict[Hashable, int] = {node: 0 for node in self._adj}
        for src, targets in self._adj.items():
            for dst in targets:
                indegree[dst] = indegree.get(dst, 0) + 1

        # Queue of nodes with zero indegree.
        zero_indeg = deque([n for n, deg in indegree.items() if deg == 0])

        order: List[Hashable] = []
        while zero_indeg:
            node = zero_indeg.popleft()
            order.append(node)
            for neighbor in self._adj.get(node, ()):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    zero_indeg.append(neighbor)

        if len(order) != len(self._adj):
            # Identify nodes involved in cycles for a clearer message.
            cycle_nodes = {n for n, deg in indegree.items() if deg > 0}
            raise CycleError(f"Graph contains a cycle involving nodes: {cycle_nodes}")

        return order


__all__ = ["DirectedGraph", "CycleError"]
