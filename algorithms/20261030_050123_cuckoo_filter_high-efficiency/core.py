import hashlib
import random
from typing import List, Any


def _int_hash(value: Any) -> int:
    """Deterministic integer hash using SHA-256."""
    h = hashlib.sha256()
    h.update(repr(value).encode('utf-8'))
    return int.from_bytes(h.digest(), 'big')


def _fingerprint(hash_val: int, size: int) -> int:
    """Return a non‑zero fingerprint of given bit size."""
    mask = (1 << size) - 1
    fp = hash_val & mask
    return fp if fp != 0 else 1  # fingerprint must be non‑zero


class CuckooFilter:
    """
    Simple cuckoo filter with deletion support.
    Parameters
    ----------
    capacity : int
        Number of buckets (will be rounded up to next power of two).
    bucket_size : int, default 4
        Number of entries per bucket.
    fingerprint_size : int, default 8
        Size of fingerprint in bits.
    max_kicks : int, default 500
        Maximum number of evictions during an insertion.
    """

    def __init__(
        self,
        capacity: int,
        bucket_size: int = 4,
        fingerprint_size: int = 8,
        max_kicks: int = 500,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        # round up to power of two for efficient modulo
        self.num_buckets = 1 << (capacity - 1).bit_length()
        self.bucket_size = bucket_size
        self.fingerprint_size = fingerprint_size
        self.max_kicks = max_kicks
        self.buckets: List[List[int]] = [[] for _ in range(self.num_buckets)]
        self.mask = self.num_buckets - 1

    def _index_hash(self, hash_val: int) -> int:
        return hash_val & self.mask

    def _alt_index(self, index: int, fingerprint: int) -> int:
        # XOR with fingerprint hash to obtain alternate bucket
        fp_hash = _int_hash(fingerprint)
        return (index ^ (fp_hash & self.mask)) & self.mask

    def _get_indices(self, item: Any) -> tuple[int, int, int]:
        h = _int_hash(item)
        fp = _fingerprint(h, self.fingerprint_size)
        i1 = self._index_hash(h)
        i2 = self._alt_index(i1, fp)
        return i1, i2, fp

    def insert(self, item: Any) -> bool:
        i1, i2, fp = self._get_indices(item)

        # Try to place fingerprint in either bucket directly
        if len(self.buckets[i1]) < self.bucket_size:
            self.buckets[i1].append(fp)
            return True
        if len(self.buckets[i2]) < self.bucket_size:
            self.buckets[i2].append(fp)
            return True

        # Eviction loop
        index = random.choice([i1, i2])
        cur_fp = fp
        for _ in range(self.max_kicks):
            bucket = self.buckets[index]
            evict_pos = random.randrange(self.bucket_size)
            # swap
            bucket[evict_pos], cur_fp = cur_fp, bucket[evict_pos]
            index = self._alt_index(index, cur_fp)
            if len(self.buckets[index]) < self.bucket_size:
                self.buckets[index].append(cur_fp)
                return True
        return False  # filter is likely full

    def contains(self, item: Any) -> bool:
        i1, i2, fp = self._get_indices(item)
        return fp in self.buckets[i1] or fp in self.buckets[i2]

    def delete(self, item: Any) -> bool:
        i1, i2, fp = self._get_indices(item)
        if fp in self.buckets[i1]:
            self.buckets[i1].remove(fp)
            return True
        if fp in self.buckets[i2]:
            self.buckets[i2].remove(fp)
            return True
        return False

    def __len__(self) -> int:
        return sum(len(b) for b in self.buckets)

    def load_factor(self) -> float:
        """Current load factor (entries / (buckets * bucket_size))."""
        total_slots = self.num_buckets * self.bucket_size
        return len(self) / total_slots
