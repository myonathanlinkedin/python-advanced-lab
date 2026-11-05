from __future__ import annotations
import heapq
from typing import Dict, List, Tuple, Set, NamedTuple, Iterable

Node = int
EdgeId = int
Demand = NamedTuple('Demand', [('src', Node), ('dst', Node), ('amount', int)])

class Edge:
    __slots__ = ('eid', 'u', 'v', 'length', 'bundles')
    def __init__(self, eid: EdgeId, u: Node, v: Node, length: float,
                 bundles: List[Tuple[int, float]]) -> None:
        """
        bundles: list of (capacity, total_cost) describing a buy‑at‑bulk contract.
        It is assumed that cost per unit is non‑increasing with capacity
        (concave cost function).
        """
        self.eid = eid
        self.u = u
        self.v = v
        self.length = length
        # store bundles sorted by cost per unit (ascending)
        self.bundles = sorted(bundles, key=lambda b: b[1] / b[0])

    def other(self, x: Node) -> Node:
        return self.v if x == self.u else self.u

    def cost(self, flow: int) -> float:
        """Return minimum cost to purchase at least `flow` units on this edge."""
        if flow <= 0:
            return 0.0
        remaining = flow
        total = 0.0
        # Greedy: use cheapest per‑unit bundles as many times as possible
        for cap, total_cost in self.bundles:
            if remaining <= 0:
                break
            # number of whole bundles we can take
            num = remaining // cap
            if num:
                total += num * total_cost
                remaining -= num * cap
        # If some flow still remains, take one more of the cheapest bundle
        if remaining > 0:
            cap, total_cost = self.bundles[0]
            total += total_cost
        return total

class Graph:
    def __init__(self) -> None:
        self._adj: Dict[Node, List[Tuple[Node, EdgeId]]] = {}
        self._edges: Dict[EdgeId, Edge] = {}
        self._next_eid: int = 0

    def add_edge(self, u: Node, v: Node, length: float,
                 bundles: List[Tuple[int, float]]) -> EdgeId:
        """Add an undirected edge and return its identifier."""
        eid = self._next_eid
        self._next_eid += 1
        edge = Edge(eid, u, v, length, bundles)
        self._edges[eid] = edge
        self._adj.setdefault(u, []).append((v, eid))
        self._adj.setdefault(v, []).append((u, eid))
        return eid

    def edges(self) -> Iterable[Edge]:
        return self._edges.values()

    def edge_by_id(self, eid: EdgeId) -> Edge:
        return self._edges[eid]

    def shortest_path(self, src: Node, dst: Node) -> List[EdgeId]:
        """Return a list of edge identifiers forming a shortest‑length path."""
        if src == dst:
            return []
        dist: Dict[Node, float] = {src: 0.0}
        prev: Dict[Node, Tuple[Node, EdgeId]] = {}
        heap: List[Tuple[float, Node]] = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d != dist[u]:
                continue
            if u == dst:
                break
            for v, eid in self._adj.get(u, []):
                nd = d + self._edges[eid].length
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = (u, eid)
                    heapq.heappush(heap, (nd, v))
        if dst not in prev:
            raise ValueError(f"No path from {src} to {dst}")
        # reconstruct path
        path: List[EdgeId] = []
        cur = dst
        while cur != src:
            pu, peid = prev[cur]
            path.append(peid)
            cur = pu
        path.reverse()
        return path

def compute_buy_at_bulk_solution(g: Graph, demands: List[Demand]) -> Tuple[Set[EdgeId], float]:
    """
    Simple heuristic:
      1. Route each demand along a shortest‑length path in the original graph.
      2. Accumulate total flow on each traversed edge.
      3. Pay the buy‑at‑bulk cost for the accumulated flow.
    Returns the set of used edge identifiers and the total cost.
    """
    flow_on_edge: Dict[EdgeId, int] = {}
    used_edges: Set[EdgeId] = set()
    for dem in demands:
        path = g.shortest_path(dem.src, dem.dst)
        for eid in path:
            used_edges.add(eid)
            flow_on_edge[eid] = flow_on_edge.get(eid, 0) + dem.amount
    total_cost = sum(g.edge_by_id(eid).cost(flow) for eid, flow in flow_on_edge.items())
    return used_edges, total_cost
