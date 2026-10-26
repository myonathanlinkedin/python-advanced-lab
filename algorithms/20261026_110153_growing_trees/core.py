"""Growing Trees: deterministic random recursive tree (RRT) generator.

A random recursive tree of order n is built by starting with a single root
node and, for each new node k (2 <= k <= n), attaching it as a child of a
uniformly random existing node. This yields a uniformly random labelled tree
on n vertices (Cayley's theorem: n^(n-2) labelled trees, each equally likely).

Complexity invariants:
  - Construction: O(n) expected time, O(n) space.
  - Depth query: O(depth) worst-case, O(log n) expected for RRT.
  - Subtree size: O(subtree_size) time, O(1) extra space (iterative).
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple


class GrowingTree:
    """A random recursive tree with deterministic, seedable construction.

    Attributes:
        n: number of nodes (1-indexed; node 1 is the root).
        parent: list where parent[i] is the parent of node i (parent[1] = 0).
        children: list of lists; children[i] holds direct children of node i.
        depth: list where depth[i] is the distance from root to node i.
    """

    def __init__(self, n: int, seed: Optional[int] = None) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self._rng = random.Random(seed)
        self.parent: List[int] = [0] * (n + 1)
        self.children: List[List[int]] = [[] for _ in range(n + 1)]
        self.depth: List[int] = [0] * (n + 1)
        self._build()

    def _build(self) -> None:
        """Attach nodes 2..n to uniformly random existing nodes. O(n)."""
        for k in range(2, self.n + 1):
            p = self._rng.randint(1, k - 1)
            self.parent[k] = p
            self.children[p].append(k)
            self.depth[k] = self.depth[p] + 1

    def is_valid(self) -> bool:
        """Verify structural invariants. O(n)."""
        if self.n < 1:
            return False
        # Root must be node 1 with no parent.
        if self.parent[1] != 0:
            return False
        # Every node 2..n has exactly one parent in [1, n-1].
        for i in range(2, self.n + 1):
            p = self.parent[i]
            if p < 1 or p >= i:
                return False
        # Children lists are consistent with parent pointers.
        for i in range(1, self.n + 1):
            for c in self.children[i]:
                if self.parent[c] != i:
                    return False
        # Depth consistency: depth[child] == depth[parent] + 1.
        for i in range(2, self.n + 1):
            if self.depth[i] != self.depth[self.parent[i]] + 1:
                return False
        # Exactly n-1 edges total.
        edge_count = sum(len(ch) for ch in self.children)
        if edge_count != self.n - 1:
            return False
        return True

    def subtree_size(self, node: int) -> int:
        """Return the number of nodes in the subtree rooted at `node`.

        Uses an iterative post-order traversal. O(subtree_size) time, O(1)
        extra space beyond the call stack (implemented iteratively).
        """
        if node < 1 or node > self.n:
            raise IndexError(f"node {node} out of range [1, {self.n}]")
        # Iterative DFS with explicit stack to avoid recursion limits.
        stack: List[Tuple[int, bool]] = [(node, False)]
        size = 0
        while stack:
            v, processed = stack.pop()
            if processed:
                size += 1
            else:
                stack.append((v, True))
                for c in self.children[v]:
                    stack.append((c, False))
        return size

    def path_to_root(self, node: int) -> List[int]:
        """Return the path from `node` up to the root (inclusive). O(depth)."""
        if node < 1 or node > self.n:
            raise IndexError(f"node {node} out of range [1, {self.n}]")
        path: List[int] = []
        cur = node
        while cur != 0:
            path.append(cur)
            cur = self.parent[cur]
        return path

    def max_depth(self) -> int:
        """Return the maximum depth in the tree. O(n)."""
        return max(self.depth[1:]) if self.n >= 1 else 0

    def leaves(self) -> List[int]:
        """Return all leaf nodes (nodes with no children). O(n)."""
        return [i for i in range(1, self.n + 1) if not self.children[i]]

    def degree(self, node: int) -> int:
        """Return the total degree (parent + children) of `node`. O(1)."""
        if node < 1 or node > self.n:
            raise IndexError(f"node {node} out of range [1, {self.n}]")
        d = len(self.children[node])
        if node != 1:
            d += 1
        return d
