from __future__ import annotations
from typing import List, Dict, Any
from custom_types import Graph


def tarjan_scc(graph: Graph) -> List[List[Any]]:
    """
    Compute the strongly connected components (SCCs) of a directed graph
    using Tarjan's algorithm.

    Returns:
        A list of SCCs, each SCC being a list of node identifiers.
        The SCCs are produced in reverse topological order (i.e., a
        component appears before any component it can reach).
    """
    index = 0                     # Global index counter
    index_map: Dict[Any, int] = {}   # Node -> discovery index
    lowlink: Dict[Any, int] = {}     # Node -> lowest reachable index
    stack: List[Any] = []            # Stack of nodes in current DFS path
    on_stack: Set[Any] = set()       # Quick membership test
    result: List[List[Any]] = []     # List of SCCs

    def strongconnect(v: Any) -> None:
        nonlocal index
        # Set the depth index for v to the smallest unused index
        index_map[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        # Consider successors of v
        for w in graph.successors(v):
            if w not in index_map:
                # Successor w has not yet been visited; recurse on it
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                # Successor w is in stack and hence in the current SCC
                lowlink[v] = min(lowlink[v], index_map[w])

        # If v is a root node, pop the stack and generate an SCC
        if lowlink[v] == index_map[v]:
            scc: List[Any] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            result.append(scc)

    # Run strongconnect for each node that has not yet been visited
    for node in graph.nodes():
        if node not in index_map:
            strongconnect(node)

    return result
