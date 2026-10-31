from __future__ import annotations

import itertools
import random
from typing import Set, List, Tuple

from custom_types import Graph, Edge, Vertex


def _is_induced_matching(edges: Set[Edge], graph: Graph) -> bool:
    """
    Verify that `edges` form an induced matching in `graph`.
    """
    # No two edges may share a vertex
    used: Set[Vertex] = set()
    for u, v in edges:
        if u in used or v in used:
            return False
        used.update((u, v))

    # No extra edges may exist between the vertices of the matching
    vertices = used
    for u in vertices:
        for v in graph.neighbours(u):
            if v in vertices and (min(u, v), max(u, v)) not in edges:
                return False
    return True


def greedy_induced_matching(graph: Graph, seed: int | None = None) -> Set[Edge]:
    """
    Simple online/greedy algorithm for induced matching.
    Edges are processed in a random order (deterministic if `seed` is given).
    The algorithm returns a maximal induced matching, which is a
    1/2‑approximation of the maximum induced matching in any graph.
    """
    rng = random.Random(seed)
    edge_list: List[Edge] = list(graph.edges())
    rng.shuffle(edge_list)

    matching: Set[Edge] = set()
    matched_vertices: Set[Vertex] = set()

    for u, v in edge_list:
        if u in matched_vertices or v in matched_vertices:
            continue
        # Check for adjacency to already matched vertices
        conflict = False
        for w in matched_vertices:
            if w in graph.neighbours(u) or w in graph.neighbours(v):
                conflict = True
                break
        if conflict:
            continue
        # Accept the edge
        matching.add((u, v))
        matched_vertices.update((u, v))

    # Defensive check (should always hold)
    assert _is_induced_matching(matching, graph)
    return matching


def brute_force_optimal_induced_matching(graph: Graph) -> Set[Edge]:
    """
    Exhaustive search for the maximum induced matching.
    Suitable only for small graphs (|V| ≤ ~12).
    Returns one optimal matching (any if multiple exist).
    """
    all_edges = list(graph.edges())
    best: Set[Edge] = set()
    # Enumerate subsets in decreasing size order for early exit
    for r in range(len(all_edges), 0, -1):
        for combo in itertools.combinations(all_edges, r):
            candidate = set(combo)
            if _is_induced_matching(candidate, graph):
                return candidate  # first found is optimal for this size
    return best  # empty matching
