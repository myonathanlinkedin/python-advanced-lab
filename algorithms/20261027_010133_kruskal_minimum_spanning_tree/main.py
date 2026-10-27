from __future__ import annotations
from typing import List

from custom_types import Edge, Graph
from engine import kruskal_mst


def _edges_from_tuples(tuples: List[tuple[int, int, int]]) -> List[Edge]:
    """Helper to build Edge objects from (u, v, weight) tuples."""
    return [Edge(weight=w, u=u, v=v) for u, v, w in tuples]


def run_tests() -> None:
    # Test 1: Simple triangle graph
    g1 = Graph(
        num_vertices=3,
        edges=_edges_from_tuples([
            (0, 1, 1),
            (1, 2, 2),
            (0, 2, 3),
        ]),
    )
    mst1, w1 = kruskal_mst(g1)
    assert w1 == 3, f"Expected weight 3, got {w1}"
    assert len(mst1) == 2, "MST of triangle must have 2 edges"
    assert set((e.u, e.v) for e in mst1) == {(0, 1), (1, 2)}

    # Test 2: Disconnected graph (two components)
    g2 = Graph(
        num_vertices=4,
        edges=_edges_from_tuples([
            (0, 1, 4),
            (2, 3, 5),
        ]),
    )
    mst2, w2 = kruskal_mst(g2)
    assert w2 == 9, f"Expected total weight 9, got {w2}"
    assert len(mst2) == 2, "Forest should contain one edge per component"
    assert set((e.u, e.v) for e in mst2) == {(0, 1), (2, 3)}

    # Test 3: Graph with parallel edges (choose lighter)
    g3 = Graph(
        num_vertices=3,
        edges=_edges_from_tuples([
            (0, 1, 10),
            (0, 1, 1),   # lighter parallel edge
            (1, 2, 2),
            (0, 2, 5),
        ]),
    )
    mst3, w3 = kruskal_mst(g3)
    assert w3 == 3, f"Expected weight 3, got {w3}"
    assert len(mst3) == 2
    assert any(e.weight == 1 and {e.u, e.v} == {0, 1} for e in mst3)

    # Test 4: Empty graph
    g4 = Graph(num_vertices=0, edges=[])
    mst4, w4 = kruskal_mst(g4)
    assert mst4 == [] and w4 == 0, "Empty graph should yield empty MST and zero weight"

    # Test 5: Single vertex, no edges
    g5 = Graph(num_vertices=1, edges=[])
    mst5, w5 = kruskal_mst(g5)
    assert mst5 == [] and w5 == 0, "Single‑vertex graph should yield empty MST"

    # Test 6: Graph containing a self‑loop (should raise on Edge creation)
    try:
        Edge(weight=1, u=0, v=0)
        assert False, "Self‑loop Edge creation must raise ValueError"
    except ValueError:
        pass

    # Test 7: Larger graph (known MST weight)
    # Classic example from CLRS (Figure 23.2)
    g7 = Graph(
        num_vertices=9,
        edges=_edges_from_tuples([
            (0, 1, 4), (0, 7, 8),
            (1, 2, 8), (1, 7, 11),
            (2, 3, 7), (2, 8, 2), (2, 5, 4),
            (3, 4, 9), (3, 5, 14),
            (4, 5, 10),
            (5, 6, 2),
            (6, 7, 1), (6, 8, 6),
            (7, 8, 7),
        ]),
    )
    mst7, w7 = kruskal_mst(g7)
    assert w7 == 37, f"Expected MST weight 37, got {w7}"
    assert len(mst7) == g7.num_vertices - 1

    print("All unit tests passed.")


if __name__ == '__main__':
    run_tests()
