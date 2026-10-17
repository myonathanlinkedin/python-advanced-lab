import hashlib
from typing import List, Tuple, Iterable, Union

Hash = bytes
ProofItem = Tuple[Hash, bool]  # (sibling_hash, sibling_is_left)


class MerkleTree:
    """Binary Merkle tree with SHA‑256 hashing."""

    def __init__(self, leaves: Iterable[Union[bytes, str]]) -> None:
        self._leaves: List[Hash] = [self._hash(l if isinstance(l, bytes) else l.encode()) for l in leaves]
        if not self._leaves:
            raise ValueError("MerkleTree requires at least one leaf")
        self._levels: List[List[Hash]] = [self._leaves]
        self._build_tree()

    @staticmethod
    def _hash(data: bytes) -> Hash:
        return hashlib.sha256(data).digest()

    @staticmethod
    def _pair_hash(left: Hash, right: Hash) -> Hash:
        return MerkleTree._hash(left + right)

    def _build_tree(self) -> None:
        current = self._leaves
        while len(current) > 1:
            next_level: List[Hash] = []
            for i in range(0, len(current), 2):
                left = current[i]
                right = current[i + 1] if i + 1 < len(current) else left
                next_level.append(self._pair_hash(left, right))
            self._levels.append(next_level)
            current = next_level

    @property
    def root(self) -> Hash:
        return self._levels[-1][0]

    def get_proof(self, index: int) -> List[ProofItem]:
        """Return Merkle proof for leaf at *index*."""
        if not (0 <= index < len(self._leaves)):
            raise IndexError("leaf index out of range")
        proof: List[ProofItem] = []
        idx = index
        for level in self._levels[:-1]:
            sibling_idx = idx ^ 1  # toggle last bit
            sibling_is_left = sibling_idx < idx
            if sibling_idx < len(level):
                sibling_hash = level[sibling_idx]
            else:
                sibling_hash = level[idx]  # duplicate when missing
            proof.append((sibling_hash, sibling_is_left))
            idx //= 2
        return proof

    @staticmethod
    def verify_proof(leaf: Union[bytes, str], proof: List[ProofItem], root: Hash) -> bool:
        """Verify that *leaf* is included in a Merkle tree with given *root*."""
        cur = MerkleTree._hash(leaf if isinstance(leaf, bytes) else leaf.encode())
        for sibling_hash, sibling_is_left in proof:
            if sibling_is_left:
                cur = MerkleTree._pair_hash(sibling_hash, cur)
            else:
                cur = MerkleTree._pair_hash(cur, sibling_hash)
        return cur == root


def _hex(b: bytes) -> str:
    return b.hex()


if __name__ == '__main__':
    # Demo data
    data = [f"leaf-{i}" for i in range(7)]

    # Build tree
    tree = MerkleTree(data)
    root_hex = _hex(tree.root)
    print(f"Merkle root: {root_hex}")

    # Verify each leaf
    for i, leaf in enumerate(data):
        proof = tree.get_proof(i)
        assert MerkleTree.verify_proof(leaf, proof, tree.root), f"Proof failed for leaf {i}"
        # Show proof hashes
        proof_hex = [(_hex(p[0]), p[1]) for p in proof]
        print(f"Leaf {i} proof: {proof_hex}")

    # Negative test: tamper with leaf
    bad_leaf = "leaf-0-modified"
    bad_proof = tree.get_proof(0)
    assert not MerkleTree.verify_proof(bad_leaf, bad_proof, tree.root), "Tampered leaf incorrectly verified"

    # Negative test: tamper with proof
    tampered_proof = list(bad_proof)
    tampered_proof[0] = (b'\x00' * 32, tampered_proof[0][1])
    assert not MerkleTree.verify_proof(data[0], tampered_proof, tree.root), "Tampered proof incorrectly verified"

    print("All tests passed.")
