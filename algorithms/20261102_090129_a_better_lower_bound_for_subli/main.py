from __future__ import annotations
import random
import math
from typing import List, Set
from custom_types import Graph
from engine import lower_bound_steiner, exact_steiner_tree_weight

def euclidean_metric(points: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            d = math.hypot(xi - xj, yi - yj)
            dist[i][j] = dist[j][i] = d
    return dist

def test_lower_bound_random_instances(num_trials: int = 200) -> None:
    random.seed(0)
    for _ in range(num_trials):
        # Small instance to keep exact DP tractable
        n = random.randint(4, 9)          # total vertices
        points = [(random.random() * 100, random.random() * 100) for _ in range(n)]
        dist = euclidean_metric(points)
        g = Graph(dist)

        # Random terminal set (at least 2 terminals)
        tsize = random.randint(2, n)
        terminals = set(random.sample(range(n), tsize))

        lb = lower_bound_steiner(g, terminals)
        opt = exact_steiner_tree_weight(g, terminals)

        # The lower bound must never exceed the optimum
        assert lb <= opt + 1e-9, f"Lower bound {lb} exceeds optimum {opt}"

        # For sanity, the bound should be at least the MST of terminals computed directly
        # (which is exactly what our function does, but we double‑check)
        # Re‑compute MST via a naive Prim to cross‑validate
        mst_weight = prim_mst_weight(g, terminals)
        assert math.isclose(lb, mst_weight, rel_tol=1e-9, abs_tol=1e-9), "LB mismatch with MST"

def prim_mst_weight(g: Graph, vertices: Set[int]) -> float:
    """Simple Prim implementation on the induced subgraph of `vertices`."""
    if len(vertices) <= 1:
        return 0.0
    visited = set()
    start = next(iter(vertices))
    visited.add(start)
    import heapq
    heap = []
    for v in vertices:
        if v != start:
            heapq.heappush(heap, (g.dist[start][v], v))
    total = 0.0
    while heap and len(visited) < len(vertices):
        w, v = heapq.heappop(heap)
        if v in visited:
            continue
        visited.add(v)
        total += w
        for u in vertices:
            if u not in visited:
                heapq.heappush(heap, (g.dist[v][u], u))
    return total

def test_edge_cases() -> None:
    # Empty terminal set
    g = Graph([[0.0]])
    assert lower_bound_steiner(g, set()) == 0.0
    # Single terminal
    assert lower_bound_steiner(g, {0}) == 0.0
    # Two terminals with known distance
    g2 = Graph([[0.0, 5.0], [5.0, 0.0]])
    assert math.isclose(lower_bound_steiner(g2, {0, 1}), 5.0, rel_tol=1e-9)

if __name__ == '__main__':
    test_edge_cases()
    test_lower_bound_random_instances()
    print("All tests passed successfully.")
