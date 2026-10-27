from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class Graph:
    """
    Simple directed graph represented by an adjacency list.
    Nodes are identified by hashable identifiers (int, str, etc.).
    """
    adjacency: Dict[object, List[object]] = field(default_factory=dict)

    def add_node(self, node: object) -> None:
        """Ensure a node exists in the adjacency structure."""
        self.adjacency.setdefault(node, [])

    def add_edge(self, src: object, dst: object) -> None:
        """Add a directed edge from src to dst, creating nodes if necessary."""
        self.add_node(src)
        self.add_node(dst)
        self.adjacency[src].append(dst)

    def nodes(self) -> Set[object]:
        """Return the set of all nodes in the graph."""
        return set(self.adjacency.keys())

    def successors(self, node: object) -> List[object]:
        """Return the list of successors of a given node."""
        return self.adjacency.get(node, [])
