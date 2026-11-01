import hashlib
import math
from typing import List, Tuple, Optional

__all__ = ["CuckooFilter"]


def _hash_bytes(key: bytes) -> bytes:
    """Return a 32-byte SHA-256 hash of the key."""
    return hashlib.sha256(key).digest()


class CuckooFilter:
    """
    A simple, deterministic implementation of a Cuckoo Filter.
    Parameters
    ----------
    capacity : int
        Expected number of items to store.
    bucket_size : int, default 4
        Number of fingerprints per bucket.
    fingerprint_size : int, default 16
        Size of each fingerprint in bits.
    max_kicks : int, default 500
        Maximum number of eviction attempts during insertion.
    """

    def __init__(
        self,
        capacity: int,
        bucket_size: int = 4,
        fingerprint_size: int = 16,
        max_kicks: int = 500,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        if fingerprint_size <= 0 or fingerprint_size > 32:
            raise ValueError("fingerprint_size must be between 1 and 32")
        if max_kicks <= 0:
            raise ValueError("max_kicks must be positive")

        self.bucket_size = bucket_size
        self.fingerprint_size = fingerprint_size
        self.max_kicks = max_kicks

        # Number of buckets: next power of two >= capacity / bucket_size
        bucket_count = 1 << (capacity - 1).bit_length()
        self.bucket_count = bucket_count
        self.bucket_mask = bucket_count - 1

        # Initialize buckets: list of lists, each bucket holds up to bucket_size fingerprints
        self.buckets: List[List[int]] = [
            [0] * bucket_size for _ in range(bucket_count)
        ]

        # Precompute fingerprint mask
        self.fp_mask = (1 << fingerprint_size) - 1

    def _fingerprint(self, key: bytes) -> int:
        h = _hash_bytes(key)
        fp = int.from_bytes(h[:2], "big") & self.fp_mask
        # Ensure fingerprint is non-zero to distinguish empty slots
        return fp if fp != 0 else 1

    def _index1(self, key: bytes) -> int:
        h = _hash_bytes(key)
        return int.from_bytes(h[2:6], "big") & self.bucket_mask

    def _index2(self, fp: int, index1: int) -> int:
        # Alternative index derived from fingerprint
        return index1 ^ (fp & self.bucket_mask)

    def _find_slot(self, bucket: List[int], fp: int) -> Optional[int]:
        for i, val in enumerate(bucket):
            if val == fp:
                return i
        return None

    def insert(self, key: bytes) -> bool:
        fp = self._fingerprint(key)
        i1 = self._index1(key)
        i2 = self._index2(fp, i1)

        # Try to insert into bucket i1
        if self._insert_into_bucket(i1, fp):
            return True
        # Try to insert into bucket i2
        if self._insert_into_bucket(i2, fp):
            return True

        # Eviction phase
        idx = i1
        for _ in range(self.max_kicks):
            slot = 0  # deterministic eviction: always evict slot 0
            evicted_fp = self.buckets[idx][slot]
            self.buckets[idx][slot] = fp
            fp = evicted_fp
            idx = self._index2(fp, idx)
            if self._insert_into_bucket(idx, fp):
                return True
        return False

    def _insert_into_bucket(self, idx: int, fp: int) -> bool:
        bucket = self.buckets[idx]
        for i in range(self.bucket_size):
            if bucket[i] == 0:
                bucket[i] = fp
                return True
        return False

    def contains(self, key: bytes) -> bool:
        fp = self._fingerprint(key)
        i1 = self._index1(key)
        i2 = self._index2(fp, i1)
        return fp in self.buckets[i1] or fp in self.buckets[i2]

    def delete(self, key: bytes) -> bool:
        fp = self._fingerprint(key)
        i1 = self._index1(key)
        i2 = self._index2(fp, i1)
        for idx in (i1, i2):
            slot = self._find_slot(self.buckets[idx], fp)
            if slot is not None:
                self.buckets[idx][slot] = 0
                return True
        return False

    def __repr__(self) -> str:
        return (
            f"CuckooFilter(capacity={self.bucket_count * self.bucket_size}, "
            f"bucket_size={self.bucket_size}, fingerprint_size={self.fingerprint_size})"
        )
