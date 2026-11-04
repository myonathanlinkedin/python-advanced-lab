"""
Merkle Tree implementation with proof generation and verification.
All operations use SHA-256 from the standard library.
"""

from __future__ import annotations

from hashlib import sha256
from typing import List, Tuple, Union, Iterable

Hash = bytes
Data = Union[bytes, str]


def _hash(data: bytes) -> Hash:
    """Return SHA-256 hash of the given bytes."""
    return sha256(data).digest()


class MerkleTree:
    """
    Merkle Tree built from a list of data items.
    Provides methods to obtain the root hash, generate proofs,
    and verify proofs against a given root.
    """

    __slots__ = ("_levels", "_root")

    def __init__(self, leaves: Iterable[Data]) -> None:
        """
        Construct a Merkle Tree from the provided leaves.
        Each leaf is hashed; if the number of leaves is odd,
        the last leaf is duplicated to form a complete binary tree.
        """
        raw_leaves: List[bytes] = [self._to_bytes(l) for l in leaves]
        if not raw_leaves:
            raise ValueError("MerkleTree requires at least one leaf")

        # Level 0: leaf hashes
        current_level: List[Hash] = [_hash(l) for l in raw_leaves]
        self._levels: List[List[Hash]] = [current_level]

        # Build upper levels
        while len(current_level) > 1:
            next_level: List[Hash] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(_hash(left + right))
            self._levels.append(next_level)
            current_level = next_level

        self._root: Hash = self._levels[-1][0]

    @staticmethod
    def _to_bytes(item: Data) -> bytes:
        """Convert a data item to bytes."""
        return item if isinstance(item, bytes) else item.encode("utf-8")

    def get_root(self) -> Hash:
        """Return the Merkle root hash."""
        return self._root

    def get_proof(self, index: int) -> List[Tuple[Hash, bool]]:
        """
        Generate a Merkle proof for the leaf at the given index.
        The proof is a list of (sibling_hash, is_left) tuples.
        is_left is True if the sibling is the left child of the parent.
        """
        if not (0 <= index < len(self._levels[0])):
            raise IndexError("Leaf index out of range")

        proof: List[Tuple[Hash, bool]] = []
        idx = index
        for level in self._levels[:-1]:
            sibling_idx = idx ^ 1  # toggle last bit to get sibling
            if sibling_idx >= len(level):
                sibling_hash = level[idx]  # duplicate last node
                is_left = False  # sibling is same as current, position irrelevant
            else:
                sibling_hash = level[sibling_idx]
                is_left = sibling_idx < idx
            proof.append((sibling_hash, is_left))
            idx //= 2
        return proof

    @staticmethod
    def verify_proof(data: Data, proof: List[Tuple[Hash, bool]], root: Hash) -> bool:
        """
        Verify that the given data, when hashed and combined with the proof,
        yields the specified root hash.
        """
        current_hash = _hash(MerkleTree._to_bytes(data))
        for sibling_hash, is_left in proof:
            if is_left:
                current_hash = _hash(sibling_hash + current_hash)
            else:
                current_hash = _hash(current_hash + sibling_hash)
        return current_hash == root

    def __repr__(self) -> str:
        return f"<MerkleTree root={self._root.hex()[:8]}...>"
