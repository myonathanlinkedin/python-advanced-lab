from __future__ import annotations

from typing import List
from custom_types import Edge, Graph
from engine import kruskal_mst

def _run_tests() -> None:
    # Test 1: Simple triangle graph
    edges1 = [
        Edge(3, 'A', 'B'),
        Edge(1, 'A', 'C'),
        Edge(2, 'B', 'C')
    ]
    mst1, weight1 = kruskal_mst(edges1)
    assert len(mst1) == 2, "Triangle MST must contain exactly 2 edges"
    assert weight1 == 3.0, f"Expected weight 3.0, got {weight1}"
    assert set(mst1) == {Edge(1, 'A', 'C'), Edge(2, 'B', 'C')}

    # Test 2: Graph with equal weights, deterministic ordering
    edges2 = [
        Edge(1, 1, 2),
        Edge(1, 1, 3),
        Edge(1, 2, 3),
        Edge(2, 3, 4)
    ]
    mst2, weight2 = kruskal_mst(edges2)
    assert len(mst2) == 3, "Should connect 4 vertices with 3 edges"
    assert weight2 == 4.0, f"Expected weight 4.0, got {weight2}"
    # The first three edges have weight 1; deterministic sorting picks (1,2), (1,3), (2,3)
    # The union of (2,3) will be skipped because 1-2 and 1-3 already connect all three vertices.
    assert mst2[0] == Edge(1, 1, 2)
    assert mst2[1] == Edge(1, 1, 3)
    assert mst2[2] == Edge(2, 3, 4)

    # Test 3: Self‑loop handling (should raise at Edge creation)
    try:
        Edge(5, 'X', 'X')
        assert False, "Self‑loop Edge creation must raise ValueError"
    except ValueError:
        pass

    # Test 4: Empty graph
    mst4, weight4 = kruskal_mst([])
    assert mst4 == [], "Empty input must yield empty MST"
    assert weight4 == 0.0, "Weight of empty MST must be 0.0"

    # Test 5: Disconnected graph (two components)
    edges5 = [
        Edge(4, 'A', 'B'),
        Edge(1, 'A', 'C'),
        Edge(3, 'D', 'E')
    ]
    mst5, weight5 = kruskal_mst(edges5)
    # Component 1: A-B-C (edges 1 and 4)
    # Component 2: D-E (edge 3)
    assert len(mst5) == 3, "Forest should contain 3 edges for 5 vertices"
    assert weight5 == 8.0, f"Expected total weight 8.0, got {weight5}"
    assert set(mst5) == {Edge(1, 'A', 'C'), Edge(4, 'A', 'B'), Edge(3, 'D', 'E')}

    # Test 6: Large deterministic test (line graph)
    n = 1000
    edges6 = [Edge(i, i, i + 1) for i in range(1, n)]
    mst6, weight6 = kruskal_mst(edges6)
    assert len(mst6) == n - 1
    assert weight6 == sum(range(1, n)), "Weight should be sum of 1..(n-1)"

    print("All unit tests passed.")


if __name__ == '__main__':
    _run_tests()
    # Demonstration: print MST of a sample graph
    sample_edges = [
        Edge(7, 'a', 'b'),
        Edge(5, 'a', 'd'),
        Edge(8, 'b', 'c'),
        Edge(9, 'b', 'd'),
        Edge(7, 'b', 'e'),
        Edge(5, 'c', 'e'),
        Edge(15, 'd', 'e'),
        Edge(6, 'd', 'f'),
        Edge(8, 'e', 'f'),
        Edge(9, 'e', 'g'),
        Edge(11, 'f', 'g')
    ]
    mst, total = kruskal_mst(sample_edges)
    print("Minimum Spanning Forest edges:")
    for e in mst:
        print(f"{e.u} -- {e.v} (weight {e.weight})")
    print(f"Total weight: {total}")
