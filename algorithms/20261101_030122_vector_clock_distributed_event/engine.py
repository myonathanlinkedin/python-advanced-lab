from __future__ import annotations

from copy import deepcopy
from typing import Dict, Mapping, MutableMapping, Optional

from .types import Clock, ComparisonResult, ProcessId


class VectorClock:
    """
    A pure‑Python implementation of a vector clock.

    The clock is represented internally as a mutable mapping from process identifiers
    to non‑negative integer counters. All operations are deterministic and side‑effect
    free except for the explicit ``increment`` method which mutates the local entry.
    """

    __slots__ = ("_pid", "_clock")

    def __init__(self, pid: ProcessId, initial: Optional[Mapping[ProcessId, int]] = None) -> None:
        """
        Create a new vector clock for the given process identifier.

        Args:
            pid: Identifier of the local process (must be hashable and comparable).
            initial: Optional initial clock state. Missing entries are treated as zero.
        """
        self._pid: ProcessId = pid
        # Use a plain dict for fast look‑ups; copy to avoid external mutation.
        self._clock: Dict[ProcessId, int] = dict(initial) if initial is not None else {}
        # Ensure the local entry exists and is non‑negative.
        self._clock.setdefault(pid, 0)
        if self._clock[pid] < 0:
            raise ValueError("Clock counters must be non‑negative integers")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    @property
    def pid(self) -> ProcessId:
        """Local process identifier."""
        return self._pid

    @property
    def clock(self) -> Mapping[ProcessId, int]:
        """Read‑only view of the underlying clock."""
        return self._clock

    def increment(self) -> None:
        """Increment the local counter to record a new event."""
        self._clock[self._pid] += 1

    def update(self, other: VectorClock) -> None:
        """
        Merge ``other`` into this clock using element‑wise maximum (the “join” operation).

        The method mutates the current clock in place.
        """
        for pid, value in other._clock.items():
            self._clock[pid] = max(self._clock.get(pid, 0), value)

    def compare(self, other: VectorClock) -> ComparisonResult:
        """
        Compare this clock with ``other`` according to the partial order defined by vector clocks.

        Returns:
            ComparisonResult.LESS       if self ≤ other and self ≠ other
            ComparisonResult.EQUAL     if self == other
            ComparisonResult.GREATER    if self ≥ other and self ≠ other
            ComparisonResult.CONCURRENT otherwise (neither ≤ nor ≥)
        """
        self_le_other = True
        self_ge_other = True

        # Union of keys from both clocks
        all_pids = set(self._clock) | set(other._clock)

        for pid in all_pids:
            a = self._clock.get(pid, 0)
            b = other._clock.get(pid, 0)
            if a < b:
                self_ge_other = False
            elif a > b:
                self_le_other = False

        if self_le_other and self_ge_other:
            return ComparisonResult.EQUAL
        if self_le_other:
            return ComparisonResult.LESS
        if self_ge_other:
            return ComparisonResult.GREATER
        return ComparisonResult.CONCURRENT

    def copy(self) -> VectorClock:
        """Return a deep copy of the vector clock."""
        return VectorClock(self._pid, deepcopy(self._clock))

    # --------------------------------------------------------------------- #
    # (De)serialization helpers
    # --------------------------------------------------------------------- #

    def to_dict(self) -> Clock:
        """Return a shallow copy suitable for JSON or other external representation."""
        return dict(self._clock)

    @classmethod
    def from_dict(cls, pid: ProcessId, data: Mapping[ProcessId, int]) -> VectorClock:
        """
        Reconstruct a VectorClock from a mapping.

        Args:
            pid: Local process identifier for the new instance.
            data: Mapping of process identifiers to counter values.
        """
        # Validation: all counters must be non‑negative integers.
        for key, value in data.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid counter for pid {key!r}: {value!r}")
        return cls(pid, data)

    # --------------------------------------------------------------------- #
    # Utility dunder methods
    # --------------------------------------------------------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self._clock == other._clock

    def __repr__(self) -> str:
        return f"VectorClock(pid={self._pid!r}, clock={self._clock!r})"
