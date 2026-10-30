import random
import itertools
import collections
import math
import cmath
from typing import Dict, Set, List, Callable, Tuple, FrozenSet

class Graph:
    """Simple undirected graph with adjacency sets."""
    def __init__(self) -> None:
        self.adj: Dict[int, Set[int]] = {}
        self._next_vertex: int = 0

    def add_vertex(self) -> int:
        v = self._next_vertex
        self.adj[v] = set()
        self._next_vertex += 1
        return v

    def add_edge(self, u: int, v: int) -> None:
        if u == v:
            raise ValueError("No self‑loops allowed")
        self.adj.setdefault(u, set()).add(v)
        self.adj.setdefault(v, set()).add(u)

    @property
    def vertices(self) -> Set[int]:
        return set(self.adj.keys())

    def neighbors(self, v: int) -> Set[int]:
        return self.adj.get(v, set())

def generate_random_regular_graph(n: int, d: int, seed: int | None = None) -> Graph:
    """
    Generates a simple d‑regular graph on n vertices using the configuration model.
    Retries until a simple graph (no loops, no multi‑edges) is obtained.
    """
    if seed is not None:
        random.seed(seed)
    if n * d % 2 != 0:
        raise ValueError("n*d must be even for a d‑regular graph")
    if d >= n:
        raise ValueError("Degree must be smaller than number of vertices")
    attempts = 0
    while True:
        attempts += 1
        stubs = list(itertools.chain.from_iterable([[v] * d for v in range(n)]))
        random.shuffle(stubs)
        g = Graph()
        for v in range(n):
            g.add_vertex()
        simple = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in g.neighbors(u):
                simple = False
                break
            g.add_edge(u, v)
        if simple:
            return g
        if attempts > 1000:
            raise RuntimeError("Failed to generate simple regular graph after many attempts")

def independence_polynomial(g: Graph) -> List[int]:
    """
    Returns the independence polynomial coefficients of graph g.
    coeffs[k] = number of independent sets of size k.
    """
    adj = g.adj
    memo: Dict[FrozenSet[int], List[int]] = {}

    def recur(verts: FrozenSet[int]) -> List[int]:
        if not verts:
            return [1]
        if verts in memo:
            return memo[verts]
        v = next(iter(verts))
        # case 1: exclude v
        without_v = recur(verts - frozenset([v]))
        # case 2: include v -> remove v and its neighbors
        neigh = adj[v] & verts
        without_Nv = recur(verts - frozenset([v]) - frozenset(neigh))
        # combine: P = without_v + x * without_Nv
        shifted = [0] + without_Nv
        max_len = max(len(without_v), len(shifted))
        res = [0] * max_len
        for i, val in enumerate(without_v):
            res[i] += val
        for i, val in enumerate(shifted):
            res[i] += val
        memo[verts] = res
        return res

    return recur(frozenset(g.vertices))

def evaluate_polynomial(coeffs: List[int], lam: complex) -> complex:
    """Evaluates polynomial Σ coeffs[k] * lam**k using Horner's method."""
    result = 0+0j
    for coeff in reversed(coeffs):
        result = result * lam + coeff
    return result

def durand_kerner(coeffs: List[int],
                  max_iter: int = 1000,
                  tol: float = 1e-12) -> List[complex]:
    """
    Finds all (complex) roots of a polynomial with given coefficients
    using the Durand–Kerner (Weierstrass) method.
    coeffs are ordered from constant term to leading coefficient.
    """
    degree = len(coeffs) - 1
    if degree <= 0:
        return []
    # Normalize to monic polynomial
    lead = coeffs[-1]
    if lead == 0:
        raise ValueError("Leading coefficient cannot be zero")
    monic = [c / lead for c in coeffs]

    # Initial guesses: points on a circle
    radius = 1.0
    roots = [radius * cmath.exp(2j * math.pi * i / degree) for i in range(degree)]

    for _ in range(max_iter):
        max_change = 0.0
        new_roots = []
        for i, r in enumerate(roots):
            # Evaluate polynomial at r
            p_val = evaluate_polynomial(monic, r)
            # Compute denominator product (r - r_j) for j != i
            denom = 1+0j
            for j, rj in enumerate(roots):
                if i != j:
                    denom *= (r - rj)
            if denom == 0:
                # Perturb slightly to avoid division by zero
                denom = 1e-12
            delta = p_val / denom
            new_r = r - delta
            max_change = max(max_change, abs(delta))
            new_roots.append(new_r)
        roots = new_roots
        if max_change < tol:
            break
    return roots

def is_zero_free(coeffs: List[int],
                 region: Callable[[complex], bool],
                 max_iter: int = 1000,
                 tol: float = 1e-12) -> bool:
    """
    Determines whether the polynomial has any root inside the given region.
    The region predicate receives a complex number and returns True if the point
    lies inside the forbidden region.
    """
    roots = durand_kerner(coeffs, max_iter=max_iter, tol=tol)
    return all(not region(r) for r in roots)
