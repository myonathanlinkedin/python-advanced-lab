from collections import deque
from typing import Dict, Generic, Iterable, List, Set, TypeVar, Tuple

TLeft = TypeVar('TLeft')
TRight = TypeVar('TRight')


class HopcroftKarp(Generic[TLeft, TRight]):
    """
    Implements the Hopcroft‑Karp algorithm for maximum cardinality matching
    in a bipartite graph G = (L ∪ R, E).

    The graph is defined by a set of left vertices L, a set of right vertices R,
    and an adjacency list from left to right vertices.
    """

    def __init__(self,
                 left: Iterable[TLeft],
                 right: Iterable[TRight],
                 edges: Iterable[Tuple[TLeft, TRight]]) -> None:
        self.left: Set[TLeft] = set(left)
        self.right: Set[TRight] = set(right)

        # adjacency list: left vertex -> list of right neighbours
        self.adj: Dict[TLeft, List[TRight]] = {u: [] for u in self.left}
        for u, v in edges:
            if u not in self.left or v not in self.right:
                raise ValueError("Edge contains vertex not in the declared partitions")
            self.adj[u].append(v)

        # matchings: left->right and right->left (None means free)
        self.pair_u: Dict[TLeft, TRight | None] = {u: None for u in self.left}
        self.pair_v: Dict[TRight, TLeft | None] = {v: None for v in self.right}
        self.dist: Dict[TLeft, int] = {}

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def maximum_matching(self) -> Dict[TLeft, TRight]:
        """
        Returns a dictionary mapping each matched left vertex to its partner
        on the right side. Unmatched left vertices are omitted.
        """
        matching_size = 0
        while self._bfs():
            for u in self.left:
                if self.pair_u[u] is None and self._dfs(u):
                    matching_size += 1
        # Build result dict without None entries
        return {u: v for u, v in self.pair_u.items() if v is not None}

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _bfs(self) -> bool:
        """
        Breadth‑first search that builds distance layers from all free left
        vertices. Returns True if there exists at least one free right vertex
        reachable via an alternating path.
        """
        queue = deque()
        for u in self.left:
            if self.pair_u[u] is None:
                self.dist[u] = 0
                queue.append(u)
            else:
                self.dist[u] = -1  # sentinel for visited but not in current layer

        found_free = False
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                pu = self.pair_v[v]
                if pu is None:
                    # An unmatched right vertex is reachable -> augmenting path exists
                    found_free = True
                elif self.dist[pu] == -1:
                    self.dist[pu] = self.dist[u] + 1
                    queue.append(pu)
        return found_free

    def _dfs(self, u: TLeft) -> bool:
        """
        Depth‑first search that attempts to find an augmenting path starting
        from left vertex u, respecting the distance layering computed by _bfs().
        """
        for v in self.adj[u]:
            pu = self.pair_v[v]
            if pu is None or (self.dist[pu] == self.dist[u] + 1 and self._dfs(pu)):
                self.pair_u[u] = v
                self.pair_v[v] = u
                return True
        # No augmenting path from u; mark as dead end for this phase
        self.dist[u] = -1
        return False
