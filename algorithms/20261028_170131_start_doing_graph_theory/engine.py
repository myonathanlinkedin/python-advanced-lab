from __future__ import annotations

from collections import deque
from heapq import heappush, heappop
from typing import Dict, List, Set, Tuple, TypeVar, Generic, Optional

from custom_types import Graph, Node

T = TypeVar('T', bound=Node)


def bfs(graph: Graph[T], start: T) -> List[T]:
    """
    Breadth‑first traversal returning nodes in visitation order.
    Raises KeyError if ``start`` is not present.
    """
    if start not in graph:
        raise KeyError(f"Start node {start!r} not in graph")
    visited: Set[T] = set()
    order: List[T] = []
    q: deque[T] = deque([start])
    visited.add(start)

    while q:
        u = q.popleft()
        order.append(u)
        for v, _ in graph.neighbours(u):
            if v not in visited:
                visited.add(v)
                q.append(v)
    return order


def dfs(graph: Graph[T], start: T) -> List[T]:
    """
    Depth‑first traversal (iterative) returning nodes in visitation order.
    Raises KeyError if ``start`` is not present.
    """
    if start not in graph:
        raise KeyError(f"Start node {start!r} not in graph")
    visited: Set[T] = set()
    order: List[T] = []
    stack: List[T] = [start]

    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        order.append(u)
        # Push neighbours in reverse order to preserve original insertion order
        for v, _ in reversed(graph.neighbours(u)):
            if v not in visited:
                stack.append(v)
    return order


def dijkstra(graph: Graph[T], source: T) -> Tuple[Dict[T, float], Dict[T, Optional[T]]]:
    """
    Dijkstra's shortest‑path algorithm for non‑negative weighted graphs.
    Returns a tuple ``(dist, prev)`` where:
      * ``dist[node]`` is the shortest distance from ``source`` to ``node`` (inf if unreachable)
      * ``prev[node]`` is the predecessor of ``node`` on the shortest path (None for source or unreachable)
    Raises KeyError if ``source`` is not present.
    """
    if source not in graph:
        raise KeyError(f"Source node {source!r} not in graph")
    dist: Dict[T, float] = {node: float('inf') for node in graph.nodes}
    prev: Dict[T, Optional[T]] = {node: None for node in graph.nodes}
    dist[source] = 0.0

    heap: List[Tuple[float, T]] = [(0.0, source)]
    while heap:
        d_u, u = heappop(heap)
        if d_u != dist[u]:
            continue  # stale entry
        for v, w in graph.neighbours(u):
            alt = d_u + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heappush(heap, (alt, v))
    return dist, prev


def topological_sort(graph: Graph[T]) -> List[T]:
    """
    Return a topological ordering of a directed acyclic graph (DAG).
    Raises ValueError if the graph contains a cycle or is undirected.
    """
    if not graph.directed:
        raise ValueError("Topological sort is defined only for directed graphs")
    # Compute in‑degree for each node
    indegree: Dict[T, int] = {node: 0 for node in graph.nodes}
    for u in graph.nodes:
        for v, _ in graph.neighbours(u):
            indegree[v] += 1

    # Collect nodes with zero in‑degree
    zero: deque[T] = deque([n for n, deg in indegree.items() if deg == 0])
    order: List[T] = []

    while zero:
        u = zero.popleft()
        order.append(u)
        for v, _ in graph.neighbours(u):
            indegree[v] -= 1
            if indegree[v] == 0:
                zero.append(v)

    if len(order) != len(graph):
        raise ValueError("Graph contains a cycle; topological ordering impossible")
    return order


def connected_components(graph: Graph[T]) -> List[Set[T]]:
    """
    Return a list of connected components.
    For directed graphs the function returns *weakly* connected components.
    """
    visited: Set[T] = set()
    components: List[Set[T]] = []

    for node in graph.nodes:
        if node in visited:
            continue
        # BFS/DFS to collect the component
        comp: Set[T] = set()
        stack: List[T] = [node]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            visited.add(u)
            for v, _ in graph.neighbours(u):
                if v not in comp:
                    stack.append(v)
            if not graph.directed:
                # For undirected graphs neighbours already cover both directions
                continue
            # For directed graphs we also need reverse edges to achieve weak connectivity
            for pred in _predecessors(graph, u):
                if pred not in comp:
                    stack.append(pred)
        components.append(comp)
    return components


def _predecessors(graph: Graph[T], target: T) -> List[T]:
    """Helper to find nodes with an edge pointing to ``target`` (used for weak connectivity)."""
    preds: List[T] = []
    for u in graph.nodes:
        for v, _ in graph.neighbours(u):
            if v == target:
                preds.append(u)
    return preds


# Export public names
__all__ = [
    "bfs",
    "dfs",
    "dijkstra",
    "topological_sort",
    "connected_components",
]
