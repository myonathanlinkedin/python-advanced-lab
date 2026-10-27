from __future__ import annotations
from typing import List, Any
from custom_types import Graph
from engine import tarjan_scc


def _assert_sccs_equal(actual: List[List[Any]], expected: List[List[Any]]) -> None:
    """
    Helper to compare SCC collections irrespective of order of components
    and order of nodes within each component.
    """
    normalize = lambda comps: sorted([sorted(comp) for comp in comps])
    assert normalize(actual) == normalize(expected), f"Expected {expected}, got {actual}"


def run_unit_tests() -> None:
    # Test 1: Empty graph
    g1 = Graph()
    _assert_sccs_equal(tarjan_scc(g1), [])

    # Test 2: Single node, no edges
    g2 = Graph()
    g2.add_node(1)
    _assert_sccs_equal(tarjan_scc(g2), [[1]])

    # Test 3: Two nodes, one direction
    g3 = Graph()
    g3.add_edge(1, 2)
    _assert_sccs_equal(tarjan_scc(g3), [[2], [1]])

    # Test 4: Simple cycle (1 -> 2 -> 3 -> 1)
    g4 = Graph()
    g4.add_edge(1, 2)
    g4.add_edge(2, 3)
    g4.add_edge(3, 1)
    _assert_sccs_equal(tarjan_scc(g4), [[1, 2, 3]])

    # Test 5: Two separate cycles
    g5 = Graph()
    for u, v in [(1, 2), (2, 1), (3, 4), (4, 3)]:
        g5.add_edge(u, v)
    _assert_sccs_equal(tarjan_scc(g5), [[1, 2], [3, 4]])

    # Test 6: Complex graph with mixed SCCs
    g6 = Graph()
    edges = [
        (1, 2), (2, 3), (3, 1),   # SCC A
        (3, 4),                   # edge from A to B
        (4, 5), (5, 6), (6, 4),   # SCC B
        (6, 7),                   # edge from B to C
        (7, 8), (8, 9), (9, 7),   # SCC C
        (9, 10)                   # edge from C to D (single node)
    ]
    for src, dst in edges:
        g6.add_edge(src, dst)
    expected_sccs = [
        [10],          # D
        [7, 8, 9],     # C
        [4, 5, 6],     # B
        [1, 2, 3]      # A
    ]
    _assert_sccs_equal(tarjan_scc(g6), expected_sccs)

    # Test 7: Self-loop
    g7 = Graph()
    g7.add_edge('a', 'a')
    g7.add_node('b')
    _assert_sccs_equal(tarjan_scc(g7), [['a'], ['b']])

    # Test 8: Disconnected nodes
    g8 = Graph()
    for n in range(5):
        g8.add_node(n)
    _assert_sccs_equal(tarjan_scc(g8), [[0], [1], [2], [3], [4]])

    print("All unit tests passed.")


def demo() -> None:
    """
    Demonstrate Tarjan's SCC detection on a sample graph and print results.
    """
    demo_graph = Graph()
    demo_edges = [
        (0, 1), (1, 2), (2, 0),   # SCC 0-1-2
        (2, 3),                   # edge to node 3
        (3, 4), (4, 5), (5, 3),   # SCC 3-4-5
        (5, 6)                    # edge to node 6 (singleton)
    ]
    for s, d in demo_edges:
        demo_graph.add_edge(s, d)

    sccs = tarjan_scc(demo_graph)
    print("Strongly Connected Components (in reverse topological order):")
    for i, comp in enumerate(sccs, 1):
        print(f"Component {i}: {sorted(comp)}")


if __name__ == '__main__':
    run_unit_tests()
    demo()
