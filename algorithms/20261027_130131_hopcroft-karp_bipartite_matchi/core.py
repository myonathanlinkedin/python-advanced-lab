from collections import deque
from typing import Dict, Generic, Hashable, Iterable, Set, Tuple, TypeVar

L = TypeVar('L', bound=Hashable)
R = TypeVar('R', bound=Hashable)


class HopcroftKarp(Generic[L, R]):
    """
    Implements the Hopcroft‑Karp algorithm for maximum cardinality matching
    in a bipartite graph G = (L ∪ R, E).  Vertices are arbitrary hashable
    objects.  The public method `maximum_matching` returns a dictionary
    mapping each left vertex to its matched right vertex (right‑to‑left
    mapping can be obtained by inverting the dict).
    """

    def __init__(self, left: Iterable[L], right: Iterable[R]) -> None:
        self.left: Set[L] = set(left)
        self.right: Set[R] = set(right)
        self.adj: Dict[L, Set[R]] = {u: set() for u in self.left}
        # internal matchings
        self.pair_u: Dict[L, R] = {}
        self.pair_v: Dict[R, L] = {}
        self.dist: Dict[L, int] = {}

    def add_edge(self, u: L, v: R) -> None:
        if u not in self.left or v not in self.right:
            raise ValueError("Both endpoints must belong to the respective partitions.")
        self.adj[u].add(v)

    # -----------------------------------------------------------------
    # Core algorithm
    # -----------------------------------------------------------------
    def _bfs(self) -> bool:
        """Breadth‑first search layering. Returns True if an augmenting path exists."""
        queue = deque()
        for u in self.left:
            if u not in self.pair_u:
                self.dist[u] = 0
                queue.append(u)
            else:
                self.dist[u] = -1  # unreachable

        found = False
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                pu = self.pair_v.get(v)
                if pu is None:
                    found = True  # free vertex on right side reached
                elif self.dist[pu] == -1:
                    self.dist[pu] = self.dist[u] + 1
                    queue.append(pu)
        return found

    def _dfs(self, u: L) -> bool:
        """Depth‑first search for augmenting paths respecting BFS layers."""
        for v in self.adj[u]:
            pu = self.pair_v.get(v)
            if pu is None or (self.dist[pu] == self.dist[u] + 1 and self._dfs(pu)):
                self.pair_u[u] = v
                self.pair_v[v] = u
                return True
        self.dist[u] = -1  # mark as dead end
        return False

    def maximum_matching(self) -> Dict[L, R]:
        """Computes a maximum matching and returns left‑to‑right mapping."""
        # Reset any previous state
        self.pair_u.clear()
        self.pair_v.clear()
        self.dist.clear()

        while self._bfs():
            for u in self.left:
                if u not in self.pair_u:
                    self._dfs(u)
        return dict(self.pair_u)
