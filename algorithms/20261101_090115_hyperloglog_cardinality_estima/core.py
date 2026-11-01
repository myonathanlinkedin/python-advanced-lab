import hashlib
import math
from typing import List


class HyperLogLog:
    """
    Production‑grade HyperLogLog cardinality estimator.

    Parameters
    ----------
    p : int
        Precision parameter (4 ≤ p ≤ 16). Number of registers m = 2**p.
    """

    __slots__ = ("p", "m", "registers", "alpha_m", "bits")

    def __init__(self, p: int = 14) -> None:
        if not (4 <= p <= 16):
            raise ValueError("Precision p must satisfy 4 ≤ p ≤ 16.")
        self.p: int = p
        self.m: int = 1 << p  # number of registers
        self.registers: List[int] = [0] * self.m
        self.bits: int = 64  # we use 64‑bit hashes
        self.alpha_m: float = self._get_alpha(self.m)

    @staticmethod
    def _get_alpha(m: int) -> float:
        """Return the bias‑correction constant α_m."""
        if m == 16:
            return 0.673
        if m == 32:
            return 0.697
        if m == 64:
            return 0.709
        return 0.7213 / (1 + 1.079 / m)

    @staticmethod
    def _hash(item) -> int:
        """Hash *item* to a 64‑bit unsigned integer using SHA‑256."""
        h = hashlib.sha256()
        h.update(repr(item).encode("utf-8"))
        # Take the first 8 bytes (64 bits) of the digest
        return int.from_bytes(h.digest()[:8], byteorder="big", signed=False)

    @staticmethod
    def _rho(w: int, max_bits: int) -> int:
        """
        Return the position of the leftmost 1-bit in *w* (1‑based).
        If *w* is zero, return max_bits + 1.
        """
        if w == 0:
            return max_bits + 1
        # Number of leading zeros in the *max_bits*‑bit representation
        leading = max_bits - w.bit_length()
        return leading + 1

    def add(self, item) -> None:
        """Incorporate *item* into the sketch."""
        x = self._hash(item)
        idx = x >> (self.bits - self.p)  # first p bits as register index
        w = x & ((1 << (self.bits - self.p)) - 1)  # remaining bits
        rank = self._rho(w, self.bits - self.p)
        if rank > self.registers[idx]:
            self.registers[idx] = rank

    def merge(self, other: "HyperLogLog") -> None:
        """Merge *other* into this sketch (in‑place)."""
        if not isinstance(other, HyperLogLog):
            raise TypeError("Can only merge with another HyperLogLog instance.")
        if self.p != other.p:
            raise ValueError("Cannot merge HyperLogLog objects with different precisions.")
        self.registers = [max(a, b) for a, b in zip(self.registers, other.registers)]

    def count(self) -> int:
        """Return the cardinality estimate."""
        # Harmonic mean of 2^{-M[i]}
        indicator = sum(2.0 ** -r for r in self.registers)
        estimate = self.alpha_m * (self.m ** 2) / indicator

        # Small range correction (Linear Counting)
        if estimate <= 5.0 / 2.0 * self.m:
            zeros = self.registers.count(0)
            if zeros != 0:
                estimate = self.m * math.log(self.m / zeros)

        # Large range correction
        elif estimate > (1.0 / 30.0) * (1 << self.bits):
            estimate = -(1 << self.bits) * math.log(1.0 - estimate / (1 << self.bits))

        return int(round(estimate))

    def __len__(self) -> int:
        """Alias for count()."""
        return self.count()

    def __repr__(self) -> str:
        return f"HyperLogLog(p={self.p}, estimate={self.count()})"
