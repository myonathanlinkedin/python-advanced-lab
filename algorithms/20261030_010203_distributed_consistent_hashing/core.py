import hashlib
import bisect
from typing import Callable, Dict, List, Tuple


def default_hash(data: bytes) -> int:
    """Deterministic hash using SHA-256, returning a 64‑bit integer."""
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big", signed=False)


class ConsistentHashRing:
    """
    Consistent hashing ring with support for virtual nodes.

    Attributes
    ----------
    replicas : int
        Number of virtual nodes per physical node (default 100).
    hash_func : Callable[[bytes], int]
        Hash function used to map keys and node identifiers onto the ring.
    """

    def __init__(self, replicas: int = 100, hash_func: Callable[[bytes], int] = None) -> None:
        if replicas <= 0:
            raise ValueError("replicas must be a positive integer")
        self.replicas: int = replicas
        self.hash_func: Callable[[bytes], int] = hash_func or default_hash

        # Mapping from hash value to node identifier
        self._ring: Dict[int, str] = {}
        # Sorted list of hash values for bisect lookup
        self._sorted_keys: List[int] = []
        # Track which virtual hashes belong to each physical node
        self._node_hashes: Dict[str, List[int]] = {}

    def __len__(self) -> int:
        """Number of physical nodes in the ring."""
        return len(self._node_hashes)

    def _hash(self, key: str) -> int:
        return self.hash_func(key.encode("utf-8"))

    def add_node(self, node_id: str, weight: int = 1) -> None:
        """
        Add a physical node with optional weight (multiple of replicas).

        Parameters
        ----------
        node_id : str
            Unique identifier for the node.
        weight : int, optional
            Relative weight; higher weight creates more virtual nodes.
        """
        if node_id in self._node_hashes:
            raise ValueError(f"Node '{node_id}' already exists")
        if weight <= 0:
            raise ValueError("weight must be a positive integer")

        total_replicas = self.replicas * weight
        hashes: List[int] = []
        for i in range(total_replicas):
            virtual_key = f"{node_id}#{i}"
            h = self._hash(virtual_key)
            # Resolve rare collisions by linear probing
            while h in self._ring:
                i += 1
                virtual_key = f"{node_id}#{i}"
                h = self._hash(virtual_key)
            self._ring[h] = node_id
            bisect.insort(self._sorted_keys, h)
            hashes.append(h)

        self._node_hashes[node_id] = hashes

    def remove_node(self, node_id: str) -> None:
        """
        Remove a physical node and all its virtual nodes.

        Parameters
        ----------
        node_id : str
            Identifier of the node to remove.
        """
        if node_id not in self._node_hashes:
            raise KeyError(f"Node '{node_id}' not found")
        for h in self._node_hashes[node_id]:
            del self._ring[h]
            idx = bisect.bisect_left(self._sorted_keys, h)
            if idx < len(self._sorted_keys) and self._sorted_keys[idx] == h:
                self._sorted_keys.pop(idx)
        del self._node_hashes[node_id]

    def get_node(self, key: str) -> str:
        """
        Return the node responsible for the given key.

        Parameters
        ----------
        key : str
            The key to locate.

        Returns
        -------
        str
            Identifier of the responsible node.

        Raises
        ------
        ValueError
            If the ring is empty.
        """
        if not self._ring:
            raise ValueError("Consistent hash ring is empty")
        h = self._hash(key)
        idx = bisect.bisect(self._sorted_keys, h) % len(self._sorted_keys)
        node_hash = self._sorted_keys[idx]
        return self._ring[node_hash]

    def get_nodes(self, key: str, n: int) -> List[str]:
        """
        Return up to `n` distinct nodes for the given key (useful for replication).

        Parameters
        ----------
        key : str
            The key to locate.
        n : int
            Number of distinct nodes desired.

        Returns
        -------
        List[str]
            List of node identifiers, ordered by proximity on the ring.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        if not self._ring:
            raise ValueError("Consistent hash ring is empty")
        result: List[str] = []
        h = self._hash(key)
        idx = bisect.bisect(self._sorted_keys, h)
        visited: set = set()
        while len(result) < n and len(visited) < len(self._node_hashes):
            real_idx = idx % len(self._sorted_keys)
            node = self._ring[self._sorted_keys[real_idx]]
            if node not in visited:
                visited.add(node)
                result.append(node)
            idx += 1
        return result
