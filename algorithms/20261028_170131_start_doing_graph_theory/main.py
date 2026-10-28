from __future__ import annotations

from typing import List, Set, Tuple, Dict

from custom_types import Graph
from engine import (
    bfs,
    dfs,
    dijkstra,
    topological_sort,
    connected_components,
)

def _assert_equal(actual, expected, msg: str = "") -> None:
    assert actual == expected, msg or f"Expected {expected!r}, got {actual!r}"

def _assert_raises(exc_type, func, *args, **kwargs) -> None:
    try:
        func(*args, **kwargs)
    except Exception as e:
        assert isinstance(e, exc_type), f"Expected exception {exc_type}, got {type(e)}"
    else:
        assert False, f"Expected exception {exc_type} but no exception was raised"

def run_tests() -> None:
    # 1. Empty graph edge cases
    empty = Graph[int](directed=False)
    _assert_raises(KeyError, bfs, empty, 0)
    _assert_raises(KeyError, dfs, empty, 0)
    _assert_raises(KeyError, dijkstra, empty, 0)

    # 2. Single node graph
    single = Graph[str](directed=True)
    single.add_node("A")
    _assert_equal(bfs(single, "A"), ["A"])
    _assert_equal(dfs(single, "A"), ["A"])
    dist, prev = dijkstra(single, "A")
    _assert_equal(dist, {"A": 0.0})
    _assert_equal(prev, {"A": None})
    _assert_equal(topological_sort(single), ["A"])
    _assert_equal(connected_components(single), [{"A"}])

    # 3. Undirected triangle graph
    tri = Graph[int](directed=False)
    tri.add_edge(1, 2)
    tri.add_edge(2, 3)
    tri.add_edge(3, 1)
    bfs_order = bfs(tri, 1)
    _assert_equal(set(bfs_order), {1, 2, 3})
    dfs_order = dfs(tri, 1)
    _assert_equal(set(dfs_order), {1, 2, 3})
    # Dijkstra on unweighted undirected graph should give distance 1 between neighbours
    dist, _ = dijkstra(tri, 1)
    _assert_equal(dist[2], 1.0)
    _assert_equal(dist[3], 1.0)
    # Connected components – single component
    _assert_equal(connected_components(tri), [{1, 2, 3}])

    # 4. Directed acyclic graph for topological sort
    dag = Graph[str](directed=True)
    dag.add_edge("A", "C")
    dag.add_edge("B", "C")
    dag.add_edge("C", "D")
    dag.add_edge("D", "E")
    order = topological_sort(dag)
    # Verify ordering respects all edges
    pos = {node: idx for idx, node in enumerate(order)}
    for u, v, _ in dag.edges:
        _assert_equal(pos[u] < pos[v], True, f"Edge {u}->{v} violates topological order")
    # Weakly connected components (should be one)
    _assert_equal(connected_components(dag), [{"A", "B", "C", "D", "E"}])

    # 5. Cycle detection in topological sort
    cyclic = Graph[int](directed=True)
    cyclic.add_edge(1, 2)
    cyclic.add_edge(2, 3)
    cyclic.add_edge(3, 1)
    _assert_raises(ValueError, topological_sort, cyclic)

    # 6. Weighted directed graph for Dijkstra
    weighted = Graph[str](directed=True)
    weighted.add_edge("S", "A", 2)
    weighted.add_edge("S", "B", 5)
    weighted.add_edge("A", "B", 1)
    weighted.add_edge("A", "C", 2)
    weighted.add_edge("B", "C", 3)
    weighted.add_edge("C", "D", 1)
    dist, prev = dijkstra(weighted, "S")
    _assert_equal(dist["D"], 5.0)  # S->A (2) + A->C (2) + C->D (1) = 5
    # Reconstruct path S -> D
    path: List[str] = []
    cur = "D"
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    _assert_equal(path, ["S", "A", "C", "D"])

    # 7. Disconnected graph – multiple components
    disc = Graph[int](directed=False)
    disc.add_edge(1, 2)
    disc.add_edge(3, 4)
    disc.add_node(5)  # isolated node
    comps = connected_components(disc)
    # Normalise component representation as frozensets for comparison
    comps_set = {frozenset(c) for c in comps}
    expected = {frozenset({1, 2}), frozenset({3, 4}), frozenset({5})}
    _assert_equal(comps_set, expected)

    print("All tests passed successfully.")


if __name__ == "__main__":
    run_tests()
