import hashlib
import bisect
from typing import Dict, List, Optional, Iterable, Tuple


class ConsistentHashRing:
    """
    A deterministic consistent hashing ring with support for virtual nodes.
    """

    _MAX_HASH = 2 ** 32

    def __init__(self, nodes: Optional[Iterable[str]] = None, replicas: int = 100) -> None:
        if replicas <= 0:
            raise ValueError("replicas must be a positive integer")
        self.replicas: int = replicas
        self._ring: Dict[int, str] = {}          # hash value -> physical node identifier
        self._sorted_keys: List[int] = []        # sorted list of hash values for bisect
        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        """
        Compute a 32‑bit hash of the given key using MD5.
        """
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16) % self._MAX_HASH

    def add_node(self, node: str) -> None:
        """
        Add a physical node and its virtual replicas to the ring.
        Adding the same node twice is ignored.
        """
        if node in self._ring.values():
            # Node already present – ignore to keep ring deterministic.
            return
        for i in range(self.replicas):
            virtual_key = f"{node}#{i}"
            h = self._hash(virtual_key)
            # Resolve unlikely collisions by linear probing.
            while h in self._ring:
                h = (h + 1) % self._MAX_HASH
            self._ring[h] = node
            bisect.insort(self._sorted_keys, h)

    def remove_node(self, node: str) -> None:
        """
        Remove a physical node and all its virtual replicas from the ring.
        Silently does nothing if the node is absent.
        """
        to_remove = [h for h, n in self._ring.items() if n == node]
        for h in to_remove:
            del self._ring[h]
            idx = bisect.bisect_left(self._sorted_keys, h)
            if idx < len(self._sorted_keys) and self._sorted_keys[idx] == h:
                self._sorted_keys.pop(idx)

    def get_node(self, key: str) -> Optional[str]:
        """
        Return the physical node responsible for the given key.
        If the ring is empty, returns None.
        """
        if not self._ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect(self._sorted_keys, h) % len(self._sorted_keys)
        node_hash = self._sorted_keys[idx]
        return self._ring[node_hash]

    def get_distribution(self, keys: Iterable[str]) -> Dict[str, List[str]]:
        """
        Map a collection of keys to their responsible nodes.
        Returns a dict: node -> list of keys.
        """
        distribution: Dict[str, List[str]] = {}
        for key in keys:
            node = self.get_node(key)
            if node is None:
                continue
            distribution.setdefault(node, []).append(key)
        return distribution

    def __len__(self) -> int:
        """Number of physical nodes in the ring."""
        return len(set(self._ring.values()))

    def __repr__(self) -> str:
        return f"<ConsistentHashRing nodes={len(self)} replicas={self.replicas}>"
