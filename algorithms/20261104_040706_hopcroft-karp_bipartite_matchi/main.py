import unittest
import random
import itertools
from core import hopcroft_karp, matching_size
from typing import Dict, List, Optional, Tuple


class HopcroftKarpTest(unittest.TestCase):
    def assertMatching(self, graph: Dict[int, List[int]], expected_size: int) -> None:
        match = hopcroft_karp(graph)
        size = sum(1 for v in match.values() if v is not None)
        self.assertEqual(size, expected_size)
        # Verify that matched pairs are consistent
        for u, v in match.items():
            if v is not None:
                self.assertIn(v, graph[u])
                # Ensure no two left vertices match to same right vertex
                self.assertEqual(
                    sum(1 for vv in match.values() if vv == v), 1
                )

    def test_empty_graph(self):
        graph: Dict[int, List[int]] = {}
        self.assertMatching(graph, 0)

    def test_no_edges(self):
        graph = {1: [], 2: []}
        self.assertMatching(graph, 0)

    def test_perfect_matching(self):
        graph = {1: [4], 2: [5], 3: [6]}
        self.assertMatching(graph, 3)

    def test_partial_matching(self):
        graph = {1: [3, 4], 2: [3], 3: []}
        self.assertMatching(graph, 2)

    def test_left_with_no_edges(self):
        graph = {1: [], 2: [3], 3: [4]}
        self.assertMatching(graph, 2)

    def test_duplicate_edges(self):
        graph = {1: [2, 2], 3: [4, 4]}
        self.assertMatching(graph, 2)

    def test_random_small_graphs(self):
        for _ in range(20):
            left = list(range(1, 5))
            right = list(range(5, 9))
            graph: Dict[int, List[int]] = {u: [] for u in left}
            for u in left:
                for v in right:
                    if random.choice([True, False]):
                        graph[u].append(v)
            # Brute force maximum matching
            best = 0
            for subset in itertools.chain.from_iterable(
                itertools.combinations(
                    [(u, v) for u in left for v in graph[u]], r
                )
                for r in range(len(left) * len(right) + 1)
            ):
                used_u: Set[int] = set()
                used_v: Set[int] = set()
                ok = True
                for u, v in subset:
                    if u in used_u or v in used_v:
                        ok = False
                        break
                    used_u.add(u)
                    used_v.add(v)
                if ok:
                    best = max(best, len(subset))
            self.assertMatching(graph, best)

    def test_consistency(self):
        graph = {1: [2, 3], 2: [3], 3: [4]}
        match = hopcroft_karp(graph)
        # Ensure mapping is a dict with all left vertices
        self.assertEqual(set(match.keys()), set(graph.keys()))
        # Ensure matched right vertices are from adjacency lists
        for u, v in match.items():
            if v is not None:
                self.assertIn(v, graph[u])

    def test_matching_size_function(self):
        graph = {1: [2], 2: [3], 3: [4]}
        size = matching_size(graph)
        self.assertEqual(size, 3)


if __name__ == "__main__":
    # Demo: compute matching for a sample graph
    sample_graph = {
        1: [2, 3],
        2: [3],
        3: [4],
        4: [5],
    }
    match = hopcroft_karp(sample_graph)
    print("Sample graph matching:")
    for u in sorted(sample_graph):
        v = match[u]
        print(f"  {u} -> {v if v is not None else 'unmatched'}")
    print(f"Matching size: {sum(1 for v in match.values() if v is not None)}")
    # Run unit tests
    unittest.main(exit=False)
