"""
core.py
=======
Reference implementation of a deterministic, pure-Python model of the
"one parser, two bugs, many routers" failure class.

The model captures the following invariants:

1. A single shared parser (``AetherFluxParser``) is used by every router
   instance.  The parser is stateful: it maintains a small ring buffer of
   recently seen tokens so that a malformed frame can poison the state for
   subsequent, well-formed frames.

2. Two distinct bugs are modelled:
   * **Bug A – Off-by-one in length field**: the parser trusts the length
     field of a frame header without clamping it to the remaining buffer.
     A crafted frame with ``length > remaining`` causes the parser to read
     past the end of the buffer, corrupting the internal token ring.
   * **Bug B – Missing state reset on error**: when the parser detects a
     checksum mismatch it discards the frame but does NOT reset the token
     ring, so the next frame is parsed against stale state.

3. A ``Router`` wraps the parser and a simple forwarding table.  Because the
   parser is shared, a single poisoned frame on one router can affect the
   parsing behaviour on *every* router that shares the same parser instance
   (the "many routers" amplification).

All operations are deterministic; no randomness, no I/O, no networking.

Time complexity:
    parse_frame   O(L)  where L = len(frame)
    forward       O(1)  (dict lookup)
    reset_state   O(K)  where K = ring buffer capacity (constant)

Space complexity:
    O(K) for the ring buffer, O(N) for the forwarding table (N = #entries).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RING_CAPACITY: int = 8          # fixed-size token ring in the parser
MAX_FRAME_LEN: int = 256        # hard cap on frame length
CHECKSUM_POLY: int = 0x07        # simple LFSR-style checksum polynomial


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    """Immutable representation of a network frame.

    Attributes
    ----------
    src : int
        Source address (0-255).
    dst : int
        Destination address (0-255).
    payload : bytes
        Arbitrary payload bytes.
    declared_len : int
        Length field as declared in the header.  May differ from
        ``len(payload)`` to model Bug A.
    checksum : int
        8-bit checksum over ``src | dst | payload``.
    """
    src: int
    dst: int
    payload: bytes
    declared_len: int
    checksum: int

    def __post_init__(self) -> None:
        if not (0 <= self.src <= 255):
            raise ValueError("src must be in [0, 255]")
        if not (0 <= self.dst <= 255):
            raise ValueError("dst must be in [0, 255]")
        if self.declared_len < 0:
            raise ValueError("declared_len must be non-negative")
        if self.checksum < 0 or self.checksum > 255:
            raise ValueError("checksum must be in [0, 255]")


@dataclass
class ForwardingEntry:
    """A single entry in a router's forwarding table."""
    dst: int
    next_hop: int
    metric: int = 1


@dataclass
class ParseResult:
    """Outcome of parsing a single frame."""
    ok: bool
    error: Optional[str] = None
    tokens: Tuple[int, ...] = ()


# ---------------------------------------------------------------------------
# Checksum helpers
# ---------------------------------------------------------------------------

def compute_checksum(src: int, dst: int, payload: bytes) -> int:
    """Compute an 8-bit LFSR-style checksum.

    Deterministic and O(len(payload)).
    """
    state = 0
    data = bytes([src, dst]) + payload
    for byte in data:
        state ^= byte
        for _ in range(8):
            msb = (state >> 7) & 1
            state = (state << 1) & 0xFF
            if msb:
                state ^= CHECKSUM_POLY
    return state & 0xFF


def make_frame(src: int, dst: int, payload: bytes,
               declared_len: Optional[int] = None,
               corrupt_checksum: bool = False) -> Frame:
    """Convenience constructor that auto-computes checksum and length."""
    if declared_len is None:
        declared_len = len(payload)
    cksum = compute_checksum(src, dst, payload)
    if corrupt_checksum:
        cksum = (cksum + 1) & 0xFF
    return Frame(src=src, dst=dst, payload=payload,
                 declared_len=declared_len, checksum=cksum)


# ---------------------------------------------------------------------------
# The shared parser (the "one parser")
# ---------------------------------------------------------------------------

class AetherFluxParser:
    """Stateful frame parser with a fixed-capacity token ring.

    Bug A: ``declared_len`` is used to slice the payload without clamping
           to the actual remaining bytes, so an oversized declared length
           causes the parser to read stale ring entries.
    Bug B: on checksum failure the ring is NOT reset, so stale tokens
           persist into the next parse.
    """

    def __init__(self, ring_capacity: int = RING_CAPACITY) -> None:
        if ring_capacity <= 0:
            raise ValueError("ring_capacity must be positive")
        self._capacity = ring_capacity
        self._ring: List[int] = [0] * ring_capacity
        self._head: int = 0          # next write position
        self._count: int = 0         # number of valid entries
        self._poisoned: bool = False

    # -- public API ---------------------------------------------------------

    def parse_frame(self, frame: Frame) -> ParseResult:
        """Parse a single frame, updating internal state.

        Returns a ``ParseResult``.  On success the token ring is updated
        with the first ``min(len(payload), capacity)`` payload bytes.
        """
        # --- Bug A: trust declared_len without clamping --------------------
        effective_len = min(frame.declared_len, MAX_FRAME_LEN)
        if effective_len > len(frame.payload):
            # Oversized declared length: read past end of payload.
            # We model this by pulling stale tokens from the ring.
            self._poisoned = True
            stale = self._read_stale(effective_len - len(frame.payload))
            tokens = tuple(frame.payload) + stale
            self._write_tokens(tokens)
            return ParseResult(ok=False, error="length_overflow",
                                tokens=tokens)

        # --- Checksum verification -----------------------------------------
        expected = compute_checksum(frame.src, frame.dst, frame.payload)
        if frame.checksum != expected:
            # Bug B: do NOT reset the ring on error.
            self._poisoned = True
            return ParseResult(ok=False, error="checksum_mismatch")

        # --- Normal path ----------------------------------------------------
        tokens = tuple(frame.payload)
        self._write_tokens(tokens)
        self._poisoned = False
        return ParseResult(ok=True, tokens=tokens)

    def reset_state(self) -> None:
        """Explicitly clear the token ring and poison flag."""
        self._ring = [0] * self._capacity
        self._head = 0
        self._count = 0
        self._poisoned = False

    @property
    def is_poisoned(self) -> bool:
        return self._poisoned

    @property
    def ring_snapshot(self) -> Tuple[int, ...]:
        """Return the current ring contents in logical order."""
        if self._count == 0:
            return ()
        start = (self._head - self._count) % self._capacity
        out: List[int] = []
        for i in range(self._count):
            out.append(self._ring[(start + i) % self._capacity])
        return tuple(out)

    # -- internal helpers ---------------------------------------------------

    def _write_tokens(self, tokens: Tuple[int, ...]) -> None:
        for tok in tokens:
            self._ring[self._head] = tok & 0xFF
            self._head = (self._head + 1) % self._capacity
            self._count = min(self._count + 1, self._capacity)

    def _read_stale(self, n: int) -> Tuple[int, ...]:
        """Read up to *n* stale tokens from the ring (Bug A helper)."""
        if self._count == 0:
            return tuple(0 for _ in range(n))
        start = (self._head - self._count) % self._capacity
        out: List[int] = []
        for i in range(n):
            out.append(self._ring[(start + i) % self._capacity])
        return tuple(out)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router:
    """A router that shares a single ``AetherFluxParser`` instance.

    Because the parser is shared, a poisoned frame on one router affects
    parsing on all routers that use the same parser (the "many routers"
    amplification).
    """

    def __init__(self, name: str, parser: AetherFluxParser,
                 table: Optional[List[ForwardingEntry]] = None) -> None:
        self.name = name
        self._parser = parser
        self._table: dict[int, ForwardingEntry] = {}
        if table:
            for entry in table:
                self._table[entry.dst] = entry

    def add_route(self, dst: int, next_hop: int, metric: int = 1) -> None:
        self._table[dst] = ForwardingEntry(dst=dst, next_hop=next_hop,
                                           metric=metric)

    def forward(self, frame: Frame) -> Optional[int]:
        """Parse *frame* and return the next-hop address, or None."""
        result = self._parser.parse_frame(frame)
        if not result.ok:
            return None
        entry = self._table.get(frame.dst)
        if entry is None:
            return None
        return entry.next_hop

    @property
    def table_size(self) -> int:
        return len(self._table)


# ---------------------------------------------------------------------------
# Multi-router topology helper
# ---------------------------------------------------------------------------

def build_topology(n_routers: int,
                   shared_parser: AetherFluxParser,
                   default_table: Optional[List[ForwardingEntry]] = None
                   ) -> List[Router]:
    """Create *n_routers* routers all sharing *shared_parser*."""
    routers: List[Router] = []
    for i in range(n_routers):
        r = Router(name=f"R{i}", parser=shared_parser, table=default_table)
        routers.append(r)
    return routers
