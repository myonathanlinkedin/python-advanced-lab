"""
core.py
Implementation of fast signed 32‑bit division using pre‑computed magic numbers.
The algorithm follows the method described in *Hacker's Delight* (Fig. 10‑2)
and the libdivide library, providing O(1) division with a single 64‑bit
multiply, an optional addition, and a shift.
"""

from __future__ import annotations
from typing import Dict, Tuple

INT_BITS = 32
INT_MIN = -2 ** (INT_BITS - 1)
INT_MAX = 2 ** (INT_BITS - 1) - 1
UINT_MAX = 2 ** INT_BITS - 1


def _to_uint32(x: int) -> int:
    """Return the low 32 bits of x as an unsigned integer."""
    return x & UINT_MAX


def _to_int32(x: int) -> int:
    """Interpret low 32 bits of x as a signed 32‑bit integer."""
    x = _to_uint32(x)
    return x - (1 << INT_BITS) if x & (1 << (INT_BITS - 1)) else x


def _clz32(x: int) -> int:
    """Count leading zeros in a 32‑bit unsigned integer."""
    # Python's bit_length gives position of highest set bit (1‑based)
    if x == 0:
        return INT_BITS
    return INT_BITS - x.bit_length()


class MagicDivider:
    """
    Holds the magic number, shift amount and sign‑adjust flag for a given divisor.
    Provides a fast division method that mimics C's trunc‑toward‑zero semantics.
    """

    __slots__ = ("d", "magic", "shift", "add")

    def __init__(self, divisor: int):
        if divisor == 0:
            raise ZeroDivisionError("division by zero")
        if divisor == 1:
            self.magic = 0
            self.shift = 0
            self.add = False
            self.d = 1
            return
        if divisor == -1:
            self.magic = 0
            self.shift = 0
            self.add = True  # add dividend before shift
            self.d = -1
            return

        self.d = divisor
        self.magic, self.shift, self.add = self._compute_magic(divisor)

    @staticmethod
    def _compute_magic(d: int) -> Tuple[int, int, bool]:
        """
        Compute magic number, shift and add‑flag for signed 32‑bit divisor d.
        Returns (magic, shift, add).
        """
        # Work with absolute value
        ad = -d if d < 0 else d
        # floor(log2(ad))
        floor_log_2 = 31 - _clz32(ad)

        # Initial values
        two31 = 1 << 31  # 2^31
        # anc = 2^31 - 1 - (2^31 mod ad)
        anc = two31 - 1 - (two31 % ad)

        # q1 = floor(2^31 / anc), r1 = 2^31 - q1*anc
        q1 = two31 // anc
        r1 = two31 - q1 * anc

        # q2 = floor(2^31 / ad), r2 = 2^31 - q2*ad
        q2 = two31 // ad
        r2 = two31 - q2 * ad

        delta = 0
        p = floor_log_2  # will be incremented before loop body
        while True:
            p += 1
            q1 <<= 1
            r1 <<= 1
            if r1 >= anc:
                q1 += 1
                r1 -= anc
            q2 <<= 1
            r2 <<= 1
            if r2 >= ad:
                q2 += 1
                r2 -= ad
            delta = ad - r2
            if q1 < delta or (q1 == delta and r1 == 0):
                continue
            break

        magic = q2 + 1
        if d < 0:
            magic = -magic
        shift = p - 32
        add = d < 0
        # Ensure magic fits in signed 32‑bit range
        magic = _to_int32(magic)
        return magic, shift, add

    def divide(self, n: int) -> int:
        """
        Fast division of signed 32‑bit integer n by the divisor used to build this
        MagicDivider. Result follows C's trunc‑toward‑zero rule.
        """
        # Emulate 32‑bit overflow on input
        n = _to_int32(n)

        if self.d == 1:
            return n
        if self.d == -1:
            # Special case: overflow for INT_MIN / -1 in C, but Python handles it.
            return -n

        # 64‑bit product, then arithmetic right shift by 32
        prod = (n * self.magic) & ((1 << 64) - 1)
        q = _to_int32(prod >> 32)  # signed high 32 bits

        if self.add:
            q += n

        # Final arithmetic shift by 'shift' bits
        if self.shift > 0:
            q = q >> self.shift
        else:
            # shift == 0, nothing to do
            pass

        # Ensure result fits in signed 32‑bit range
        return _to_int32(q)

    def remainder(self, n: int) -> int:
        """Compute n % d using the fast division result."""
        q = self.divide(n)
        return _to_int32(n - q * self.d)


# Cache MagicDivider instances per divisor for reuse
_divider_cache: Dict[int, MagicDivider] = {}


def fast_divide(n: int, d: int) -> int:
    """
    Public API: fast signed 32‑bit division.
    Mirrors C's trunc‑toward‑zero semantics.
    """
    if d not in _divider_cache:
        _divider_cache[d] = MagicDivider(d)
    return _divider_cache[d].divide(n)


def fast_mod(n: int, d: int) -> int:
    """Public API: fast signed 32‑bit remainder."""
    if d not in _divider_cache:
        _divider_cache[d] = MagicDivider(d)
    return _divider_cache[d].remainder(n)
