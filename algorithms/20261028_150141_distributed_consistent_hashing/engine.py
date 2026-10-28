import bisect
import hashlib
from typing import Dict, List, Tuple

from custom_types import Node

__all__ = ["ConsistentHashRouter"]

class ConsistentHashRouter:
    """
    Implements a consistent hashing router with virtual nodes.
    """

    def __init__(self, virtual_nodes_per_node: int = 100) -> None:
        if virtual_nodes_per_node <= 0:
            raise ValueError("virtual_nodes_per_node must be positive")
        self._vn_per_node: int = virtual_nodes_per_node
        self._ring: List[int] = []  # Sorted list of virtual node hashes
        self._hash_to_node: Dict[int, Node] = {}
        self._node_to_hashes: Dict[Node, List[int]] = {}

    @staticmethod
    def _hash(value: str) -> int:
        """
        Deterministic hash function using SHA-256.
        """
        return int.from_bytes(hashlib.sha256(value.encode()).digest(), "big")

    def add_node(self, node: Node) -> None:
        """
        Adds a physical node and its virtual nodes to the ring.
        """
        if node in self._node_to_hashes:
            raise ValueError(f"Node {node.id} already exists")
        hashes: List[int] = []
        for i in range(self._vn_per_node):
            vnode_key = f"{node.id}#{i}"
            h = self._hash(vnode_key)
            bisect.insort(self._ring, h)
            self._hash_to_node[h] = node
            hashes.append(h)
        self._node_to_hashes[node] = hashes

    def remove_node(self, node: Node) -> None:
        """
        Removes a physical node and all its virtual nodes from the ring.
        """
        if node not in self._node_to_hashes:
            raise ValueError(f"Node {node.id} does not exist")
        for h in self._node_to_hashes[node]:
            idx = bisect.bisect_left(self._ring, h)
            if idx < len(self._ring) and self._ring[idx] == h:
                self._ring.pop(idx)
                del self._hash_to_node[h]
        del self._node_to_hashes[node]

    def get_node(self, key: str) -> Node:
        """
        Returns the node responsible for the given key.
        """
        if not self._ring:
            raise RuntimeError("No nodes in the ring")
        h = self._hash(key)
        idx = bisect.bisect_right(self._ring, h)
        if idx == len(self._ring):
            idx = 0  # wrap around
        return self._hash_to_node[self._ring[idx]]
