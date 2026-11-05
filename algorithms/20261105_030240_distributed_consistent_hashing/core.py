import hashlib
import bisect
from typing import List, Tuple, Dict, Iterable


class ConsistentHashRing:
    """
    Consistent hashing ring with virtual nodes.

    Attributes
    ----------
    replicas : int
        Number of virtual nodes per physical node.
    _ring : List[int]
        Sorted list of hash values (the ring positions).
    _hash_node_map : Dict[int, str]
        Mapping from ring position to physical node identifier.
    _node_hashes : Dict[str, List[int]]
        Mapping from physical node identifier to its virtual node hashes.
    """

    def __init__(self, replicas: int = 100) -> None:
        if replicas <= 0:
            raise ValueError("replicas must be positive")
        self.replicas: int = replicas
        self._ring: List[int] = []
        self._hash_node_map: Dict[int, str] = {}
        self._node_hashes: Dict[str, List[int]] = {}

    @staticmethod
    def _hash(value: str) -> int:
        """Deterministic 32‑bit hash using MD5."""
        h = hashlib.md5(value.encode('utf-8')).hexdigest()
        return int(h[:8], 16)  # use first 8 hex digits → 32 bits

    def add_node(self, node_id: str) -> None:
        """Add a physical node and its virtual replicas to the ring."""
        if node_id in self._node_hashes:
            raise ValueError(f"Node {node_id!r} already exists")
        hashes: List[int] = []
        for i in range(self.replicas):
            virtual_id = f"{node_id}#{i}"
            h = self._hash(virtual_id)
            # Resolve collisions by linear probing (unlikely with 32‑bit space)
            while h in self._hash_node_map:
                h = (h + 1) % (2 ** 32)
            bisect.insort(self._ring, h)
            self._hash_node_map[h] = node_id
            hashes.append(h)
        self._node_hashes[node_id] = hashes

    def remove_node(self, node_id: str) -> None:
        """Remove a physical node and all its virtual replicas."""
        if node_id not in self._node_hashes:
            raise KeyError(f"Node {node_id!r} not found")
        for h in self._node_hashes[node_id]:
            idx = bisect.bisect_left(self._ring, h)
            if idx < len(self._ring) and self._ring[idx] == h:
                self._ring.pop(idx)
                del self._hash_node_map[h]
        del self._node_hashes[node_id]

    def get_node(self, key: str) -> str:
        """Return the physical node responsible for the given key."""
        if not self._ring:
            raise RuntimeError("Hash ring is empty")
        h = self._hash(key)
        idx = bisect.bisect_right(self._ring, h) % len(self._ring)
        node_hash = self._ring[idx]
        return self._hash_node_map[node_hash]

    def get_nodes(self, key: str, count: int) -> List[str]:
        """
        Return up to `count` distinct nodes for the given key,
        useful for replication.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        if not self._ring:
            raise RuntimeError("Hash ring is empty")
        result: List[str] = []
        h = self._hash(key)
        idx = bisect.bisect_right(self._ring, h)
        visited: set = set()
        ring_len = len(self._ring)
        while len(result) < min(count, len(self._node_hashes)):
            node_hash = self._ring[idx % ring_len]
            node = self._hash_node_map[node_hash]
            if node not in visited:
                visited.add(node)
                result.append(node)
            idx += 1
        return result

    def nodes(self) -> List[str]:
        """Return a list of all physical nodes currently in the ring."""
        return list(self._node_hashes.keys())

    def __len__(self) -> int:
        return len(self._node_hashes)

    def __repr__(self) -> str:
        return f"<ConsistentHashRing replicas={self.replicas} nodes={self.nodes()}>"
