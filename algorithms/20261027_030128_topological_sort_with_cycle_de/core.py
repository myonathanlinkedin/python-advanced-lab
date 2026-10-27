from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Set


class CycleError(Exception):
    """Raised when a directed graph contains a cycle."""

    def __init__(self, cycle_nodes: Set[Any]) -> None:
        self.cycle_nodes: Set[Any] = cycle_nodes
        message = f"Graph contains a cycle involving nodes: {sorted(cycle_nodes)}"
        super().__init__(message)


def topological_sort(graph: Dict[Any, Iterable[Any]]) -> List[Any]:
    """
    Perform a topological sort on a directed graph.

    Parameters
    ----------
    graph : Dict[Any, Iterable[Any]]
        Mapping from a node to an iterable of its outgoing neighbours.

    Returns
    -------
    List[Any]
        A list of nodes in topological order.

    Raises
    ------
    CycleError
        If the graph contains at least one directed cycle.
    """
    # Build the full set of nodes (including isolated ones)
    nodes: Set[Any] = set(graph.keys())
    for neighbours in graph.values():
        nodes.update(neighbours)

    # Initialise indegree counts
    indegree: Dict[Any, int] = {node: 0 for node in nodes}
    adjacency: Dict[Any, List[Any]] = defaultdict(list)

    for src, neighbours in graph.items():
        for dst in neighbours:
            adjacency[src].append(dst)
            indegree[dst] += 1

    # Queue of nodes with zero indegree
    zero_indeg = deque([n for n in nodes if indegree[n] == 0])

    order: List[Any] = []
    while zero_indeg:
        node = zero_indeg.popleft()
        order.append(node)
        for neighbour in adjacency.get(node, []):
            indegree[neighbour] -= 1
            if indegree[neighbour] == 0:
                zero_indeg.append(neighbour)

    if len(order) != len(nodes):
        # Nodes not visited are part of a cycle
        remaining = {n for n in nodes if indegree[n] > 0}
        raise CycleError(remaining)

    return order
