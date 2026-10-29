"""
HyperLogLog (HLL) cardinality estimator.

Reference implementation adhering to the original Flajolet et al. algorithm
with small and large range corrections. Uses only the Python standard library.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Iterable


def _hash64(value: object) -> int:
    """
    Deterministic 64‑bit hash using SHA‑256.
    The first 8 bytes of the digest are interpreted as a big‑endian integer.
    """
    h = hashlib.sha256()
    h.update(repr(value).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big", signed=False)


def _rho(w: int, max_bits: int) -> int:
    """
    Return the position of the leftmost 1-bit in ``w`` (1‑based).
    If ``w`` is zero, returns ``max_bits + 1``.
    """
    if w == 0:
        return max_bits + 1
    leading_zeros = max_bits - w.bit_length()
    return leading_zeros + 1


class HyperLogLog:
    """
    HyperLogLog estimator.

    Parameters
    ----------
    p : int, optional
        Precision (number of bits used for the register index). Must satisfy
        4 ≤ p ≤ 16. Default is 14, giving 16384 registers and ~1.04% standard error.
    """

    __slots__ = ("p", "m", "registers", "alpha_m")

    def __init__(self, p: int = 14) -> None:
        if not (4 <= p <= 16):
            raise ValueError("Precision p must be in the range [4, 16].")
        self.p: int = p
        self.m: int = 1 << p  # number of registers
        self.registers: List[int] = [0] * self.m
        # Alpha constant depends on m
        if self.m == 16:
            self.alpha_m = 0.673
        elif self.m == 32:
            self.alpha_m = 0.697
        elif self.m == 64:
            self.alpha_m = 0.709
        else:
            self.alpha_m = 0.7213 / (1 + 1.079 / self.m)

    def _register_index_and_w(self, x: int) -> tuple[int, int]:
        """
        Split the 64‑bit hash ``x`` into register index (first p bits)
        and the remaining bits ``w``.
        """
        idx = x >> (64 - self.p)
        w = x & ((1 << (64 - self.p)) - 1)
        return idx, w

    def add(self, item: object) -> None:
        """
        Incorporate ``item`` into the estimator.
        """
        x = _hash64(item)
        idx, w = self._register_index_and_w(x)
        rank = _rho(w, 64 - self.p)
        if rank > self.registers[idx]:
            self.registers[idx] = rank

    def update(self, items: Iterable[object]) -> None:
        """
        Add a collection of items.
        """
        for item in items:
            self.add(item)

    def merge(self, other: "HyperLogLog") -> None:
        """
        Merge another HLL into this one. Both must share the same precision.
        """
        if self.p != other.p:
            raise ValueError("Cannot merge HLLs with different precisions.")
        self.registers = [max(a, b) for a, b in zip(self.registers, other.registers)]

    def count(self) -> int:
        """
        Return the estimated cardinality as an integer.
        """
        # Harmonic mean of 2^{-register}
        indicator = sum(2.0 ** -r for r in self.registers)
        raw_estimate = self.alpha_m * (self.m ** 2) / indicator

        # Small range correction (linear counting)
        V = self.registers.count(0)
        if V != 0 and raw_estimate <= 5.0 / 2.0 * self.m:
            estimate = self.m * math.log(self.m / V)
        else:
            estimate = raw_estimate

        # Large range correction
        if estimate > (1.0 / 30.0) * (1 << 64):
            estimate = -(1 << 64) * math.log(1.0 - estimate / (1 << 64))

        return int(round(estimate))

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"HyperLogLog(p={self.p}, estimate={self.count()})"
