from __future__ import annotations
from typing import Dict, Any


class GCounter:
    """
    Grow-only counter (G-Counter) used as a building block for PN-Counter.
    Stores a non‑negative integer per replica identifier.
    """

    __slots__ = ("_state",)

    def __init__(self) -> None:
        self._state: Dict[str, int] = {}

    def increment(self, replica_id: str, amount: int = 1) -> None:
        """Increase the counter for *replica_id* by *amount* (must be > 0)."""
        if amount <= 0:
            raise ValueError("increment amount must be positive")
        self._state[replica_id] = self._state.get(replica_id, 0) + amount

    def value(self) -> int:
        """Return the total sum of all replica contributions."""
        return sum(self._state.values())

    def merge(self, other: GCounter) -> None:
        """
        Merge another GCounter into this one.
        For each replica identifier the maximum observed value is kept.
        """
        for rid, val in other._state.items():
            self._state[rid] = max(self._state.get(rid, 0), val)

    def to_dict(self) -> Dict[str, int]:
        """Return a shallow copy of the internal state (useful for testing)."""
        return dict(self._state)

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> GCounter:
        """Create a GCounter from a raw dictionary."""
        obj = cls()
        for rid, val in data.items():
            if val < 0:
                raise ValueError("GCounter values must be non‑negative")
            obj._state[rid] = int(val)
        return obj

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, GCounter) and self._state == other._state

    def __repr__(self) -> str:
        return f"GCounter({self._state})"


class PNCounter:
    """
    Positive‑Negative Counter (PN-Counter) CRDT.
    Internally composed of two G‑Counters:
        * p_counter – records increments
        * n_counter – records decrements
    The observable value is p_counter – n_counter.
    """

    __slots__ = ("_p", "_n")

    def __init__(self) -> None:
        self._p = GCounter()
        self._n = GCounter()

    # --------------------------------------------------------------------- #
    # Mutating operations
    # --------------------------------------------------------------------- #
    def increment(self, replica_id: str, amount: int = 1) -> None:
        """Record an increment for *replica_id*."""
        self._p.increment(replica_id, amount)

    def decrement(self, replica_id: str, amount: int = 1) -> None:
        """Record a decrement for *replica_id*."""
        if amount <= 0:
            raise ValueError("decrement amount must be positive")
        self._n.increment(replica_id, amount)

    # --------------------------------------------------------------------- #
    # Query operations
    # --------------------------------------------------------------------- #
    def value(self) -> int:
        """Current logical value of the counter."""
        return self._p.value() - self._n.value()

    # --------------------------------------------------------------------- #
    # Merge operation
    # --------------------------------------------------------------------- #
    def merge(self, other: PNCounter) -> None:
        """
        Merge another PN‑Counter into this one.
        The operation is commutative, associative and idempotent.
        """
        self._p.merge(other._p)
        self._n.merge(other._n)

    # --------------------------------------------------------------------- #
    # Utility / introspection
    # --------------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Dict[str, int]]:
        """Serialise the full state for persistence or testing."""
        return {"p": self._p.to_dict(), "n": self._n.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, int]]) -> PNCounter:
        """Recreate a PN‑Counter from a dictionary produced by :meth:`to_dict`."""
        obj = cls()
        obj._p = GCounter.from_dict(data.get("p", {}))
        obj._n = GCounter.from_dict(data.get("n", {}))
        return obj

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, PNCounter)
            and self._p == other._p
            and self._n == other._n
        )

    def __repr__(self) -> str:
        return f"PNCounter(p={self._p}, n={self._n})"
