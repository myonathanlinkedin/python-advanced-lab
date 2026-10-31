from __future__ import annotations

import random
import sys
from typing import Set

from custom_types import Graph, Edge
from engine import greedy_induced_matching, brute_force_optimal_induced_matching

def generate_random_graph(n: int, p: float, seed: int | None = None) -> Graph:
    """
    Generate an Erdős–Rényi G(n, p) random undirected graph without self‑loops.
    """
    rng = random.Random(seed)
    vertices = list(range(n))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return Graph.from_edges(vertices, edges)


def _assert_half_optimal(g: Graph, seed: int | None = None) -> None:
    """
    Verify that the greedy algorithm achieves at least half of the optimal size.
    """
    greedy = greedy_induced_matching(g, seed=seed)
    optimal = brute_force_optimal_induced_matching(g)
    # Edge case: optimal may be empty
    if len(optimal) == 0:
        assert len(greedy) == 0, "Greedy should return empty matching on empty optimal"
    else:
        ratio = len(greedy) / len(optimal)
        assert ratio >= 0.5, f"Approximation ratio {ratio:.3f} < 0.5"


def run_unit_tests() -> None:
    # Test 1: Empty graph
    g0 = Graph.empty()
    assert greedy_induced_matching(g0) == set()
    assert brute_force_optimal_induced_matching(g0) == set()

    # Test 2: Single edge
    g1 = Graph.from_edges([0, 1], [(0, 1)])
    assert greedy_induced_matching(g1) == {(0, 1)}
    assert brute_force_optimal_induced_matching(g1) == {(0, 1)}

    # Test 3: Path of length 3 (4 vertices, edges 0-1,1-2,2-3)
    edges_path = [(0, 1), (1, 2), (2, 3)]
    g2 = Graph.from_edges(range(4), edges_path)
    greedy = greedy_induced_matching(g2, seed=42)
    optimal = brute_force_optimal_induced_matching(g2)
    assert len(optimal) == 2  # edges (0,1) and (2,3) form an induced matching
    assert len(greedy) >= 1  # greedy may pick one of the two edges
    _assert_half_optimal(g2, seed=42)

    # Test 4: Small random graphs (exhaustive verification)
    rng = random.Random(12345)
    for n in range(2, 9):  # up to 8 vertices (manageable brute force)
        for _ in range(20):
            p = rng.random() * 0.7  # keep density moderate
            g = generate_random_graph(n, p, seed=rng.randint(0, 1 << 30))
            _assert_half_optimal(g, seed=0)  # deterministic greedy

    # Test 5: Low‑degree graph (star)
    star_edges = [(0, i) for i in range(1, 6)]
    g_star = Graph.from_edges(range(6), star_edges)
    greedy_star = greedy_induced_matching(g_star, seed=7)
    optimal_star = brute_force_optimal_induced_matching(g_star)
    assert len(optimal_star) == 1
    assert len(greedy_star) == 1
    _assert_half_optimal(g_star, seed=7)

    print("All unit tests passed.")


def demo() -> None:
    """
    Demonstration on a medium‑sized random graph.
    """
    n = 15
    p = 0.3
    seed = 2023
    g = generate_random_graph(n, p, seed=seed)
    greedy = greedy_induced_matching(g, seed=seed)
    optimal = brute_force_optimal_induced_matching(g)  # feasible for n=15 in many cases
    print(f"Graph: {n} vertices, edge probability {p}")
    print(f"Greedy induced matching size: {len(greedy)}")
    print(f"Optimal induced matching size (brute‑force): {len(optimal)}")
    if len(optimal) > 0:
        print(f"Approximation ratio: {len(greedy) / len(optimal):.3f}")


if __name__ == "__main__":
    # Run exhaustive unit tests first
    run_unit_tests()
    # Then show a short demo
    demo()
