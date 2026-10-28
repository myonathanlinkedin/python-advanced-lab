from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generic, Hashable, List, Tuple, TypeVar, Iterable, Set

Node = TypeVar('Node', bound=Hashable)


@dataclass
class Graph(Generic[Node]):
    """
    Simple adjacency‑list graph representation.
    Supports directed and undirected graphs with optional edge weights.
    """
    directed: bool = False
    _adj: Dict[Node, List[Tuple[Node, float]]] = field(default_factory=dict, init=False, repr=False)

    # --------------------------------------------------------------------- #
    # Mutating API
    # --------------------------------------------------------------------- #
    def add_node(self, node: Node) -> None:
        """Add a node to the graph. No effect if the node already exists."""
        self._adj.setdefault(node, [])

    def add_edge(self, u: Node, v: Node, weight: float = 1.0) -> None:
        """
        Add an edge from ``u`` to ``v`` with the given weight.
        For undirected graphs a symmetric edge is added automatically.
        """
        if weight < 0:
            raise ValueError("Edge weight must be non‑negative")
        self.add_node(u)
        self.add_node(v)
        self._adj[u].append((v, weight))
        if not self.directed:
            self._adj[v].append((u, weight))

    # --------------------------------------------------------------------- #
    # Inspection API
    # --------------------------------------------------------------------- #
    @property
    def nodes(self) -> Set[Node]:
        """Return a set of all nodes present in the graph."""
        return set(self._adj.keys())

    @property
    def edges(self) -> List[Tuple[Node, Node, float]]:
        """Return a list of edges as (source, target, weight)."""
        edge_list: List[Tuple[Node, Node, float]] = []
        for u, neighbours in self._adj.items():
            for v, w in neighbours:
                if self.directed or (u <= v):  # type: ignore[operator]
                    edge_list.append((u, v, w))
        return edge_list

    def neighbours(self, node: Node) -> List[Tuple[Node, float]]:
        """Return a list of (neighbour, weight) pairs for ``node``."""
        if node not in self._adj:
            raise KeyError(f"Node {node!r} not in graph")
        return list(self._adj[node])

    def __contains__(self, node: Node) -> bool:
        return node in self._adj

    def __len__(self) -> int:
        return len(self._adj)

    def __repr__(self) -> str:
        kind = "Directed" if self.directed else "Undirected"
        return f"{kind}Graph(nodes={len(self)}, edges={len(self.edges)})"


# Export public names
__all__ = ["Graph", "Node"]
