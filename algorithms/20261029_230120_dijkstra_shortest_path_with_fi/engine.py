from __future__ import annotations
from typing import Dict, List, Tuple, Callable, Optional, Any
import math
from custom_types import FibHeapNode, K, V

# ---------- Fibonacci Heap Implementation ----------
class FibonacciHeap(Generic[K, V]):
    """
    A min-oriented Fibonacci Heap.
    Supports insert, find_min, extract_min, decrease_key, and is_empty.
    """
    def __init__(self) -> None:
        self.min_node: Optional[FibHeapNode[K, V]] = None
        self.total_nodes: int = 0
        # map values to nodes for O(1) decrease_key access
        self._node_map: Dict[V, FibHeapNode[K, V]] = {}

    # ----- internal helpers -----
    def _link(self, y: FibHeapNode[K, V], x: FibHeapNode[K, V]) -> None:
        """Make node y a child of node x."""
        # remove y from root list
        y.left.right = y.right
        y.right.left = y.left

        # make y a child of x
        y.parent = x
        if x.child is None:
            y.left = y.right = y
            x.child = y
        else:
            # insert y into x's child circular list
            y.left = x.child
            y.right = x.child.right
            x.child.right.left = y
            x.child.right = y
        x.degree += 1
        y.mark = False

    def _consolidate(self) -> None:
        """Consolidate trees in the root list so that there is at most one tree of any degree."""
        max_degree = int(math.log2(self.total_nodes)) + 1 if self.total_nodes > 0 else 0
        A: List[Optional[FibHeapNode[K, V]]] = [None] * (max_degree + 1)

        # Gather root nodes into a list to avoid mutation during iteration
        roots: List[FibHeapNode[K, V]] = []
        x = self.min_node
        if x is not None:
            while True:
                roots.append(x)
                x = x.right
                if x is self.min_node:
                    break

        for w in roots:
            x = w
            d = x.degree
            while A[d] is not None:
                y = A[d]
                if x.key > y.key:
                    x, y = y, x
                self._link(y, x)
                A[d] = None
                d += 1
            A[d] = x

        # Reconstruct the root list and find new min
        self.min_node = None
        for node in A:
            if node is not None:
                if self.min_node is None:
                    node.left = node.right = node
                    self.min_node = node
                else:
                    # insert node into root list
                    node.left = self.min_node
                    node.right = self.min_node.right
                    self.min_node.right.left = node
                    self.min_node.right = node
                    if node.key < self.min_node.key:
                        self.min_node = node

    # ----- public API -----
    def is_empty(self) -> bool:
        return self.min_node is None

    def insert(self, key: K, value: V) -> None:
        """Insert a new (key, value) pair."""
        node = FibHeapNode(key, value)
        # add to root list
        if self.min_node is None:
            self.min_node = node
        else:
            node.left = self.min_node
            node.right = self.min_node.right
            self.min_node.right.left = node
            self.min_node.right = node
            if node.key < self.min_node.key:
                self.min_node = node
        self.total_nodes += 1
        self._node_map[value] = node

    def find_min(self) -> Optional[Tuple[K, V]]:
        if self.min_node is None:
            return None
        return self.min_node.key, self.min_node.value

    def extract_min(self) -> Optional[Tuple[K, V]]:
        """Remove and return the node with minimum key."""
        z = self.min_node
        if z is not None:
            # For each child of z, add it to the root list
            if z.child is not None:
                children = []
                x = z.child
                while True:
                    children.append(x)
                    x = x.right
                    if x is z.child:
                        break
                for x in children:
                    # remove parent reference
                    x.parent = None
                    # add x to root list
                    x.left = self.min_node
                    x.right = self.min_node.right
                    self.min_node.right.left = x
                    self.min_node.right = x
            # remove z from root list
            z.left.right = z.right
            z.right.left = z.left
            if z is z.right:
                self.min_node = None
            else:
                self.min_node = z.right
                self._consolidate()
            self.total_nodes -= 1
            # cleanup map
            self._node_map.pop(z.value, None)
            return z.key, z.value
        return None

    def decrease_key(self, value: V, new_key: K) -> None:
        """Decrease the key of the node containing `value` to `new_key`."""
        node = self._node_map.get(value)
        if node is None:
            raise KeyError(f'Value {value!r} not found in heap')
        if new_key > node.key:
            raise ValueError('new key is greater than current key')
        node.key = new_key
        y = node.parent
        if y is not None and node.key < y.key:
            self._cut(node, y)
            self._cascading_cut(y)
        if self.min_node is None or node.key < self.min_node.key:
            self.min_node = node

    def _cut(self, x: FibHeapNode[K, V], y: FibHeapNode[K, V]) -> None:
        """Remove x from the child list of y and add x to the root list."""
        # remove x from child list of y
        if x.right == x:
            y.child = None
        else:
            x.right.left = x.left
            x.left.right = x.right
            if y.child is x:
                y.child = x.right
        y.degree -= 1
        # add x to root list
        x.left = self.min_node
        x.right = self.min_node.right
        self.min_node.right.left = x
        self.min_node.right = x
        x.parent = None
        x.mark = False

    def _cascading_cut(self, y: FibHeapNode[K, V]) -> None:
        """Perform a cascading cut on node y."""
        z = y.parent
        if z is not None:
            if not y.mark:
                y.mark = True
            else:
                self._cut(y, z)
                self._cascading_cut(z)

    # ----- utility for testing -----
    def __len__(self) -> int:
        return self.total_nodes

# ---------- Dijkstra's Algorithm ----------
def dijkstra(
    graph: Dict[int, List[Tuple[int, float]]],
    source: int
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """
    Compute shortest path distances from `source` to all reachable vertices
    using a Fibonacci heap priority queue.

    Returns:
        distances: mapping vertex -> shortest distance (float('inf') if unreachable)
        predecessors: mapping vertex -> previous vertex on the shortest path (None for source or unreachable)
    """
    # Initialize distances and predecessor maps
    dist: Dict[int, float] = {v: math.inf for v in graph}
    prev: Dict[int, Optional[int]] = {v: None for v in graph}
    if source not in graph:
        raise ValueError('Source vertex not in graph')
    dist[source] = 0.0

    # Initialize heap and insert source
    heap = FibonacciHeap[float, int]()
    heap.insert(0.0, source)

    while not heap.is_empty():
        current_key, u = heap.extract_min()  # type: ignore
        # current_key is the finalized shortest distance to u
        for v, weight in graph.get(u, []):
            if weight < 0:
                raise ValueError('Graph contains negative edge weight')
            alt = dist[u] + weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                if v in heap._node_map:
                    heap.decrease_key(v, alt)
                else:
                    heap.insert(alt, v)

    return dist, prev
