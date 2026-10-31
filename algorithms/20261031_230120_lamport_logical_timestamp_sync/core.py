from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True, slots=True)
class Message:
    """Immutable Lamport message carrying a logical timestamp."""
    sender: int
    receiver: int
    timestamp: int


class LamportClockEngine:
    """
    Engine managing Lamport logical clocks for a fixed set of processes.

    Each process is identified by an integer `pid`. The engine tracks a
    monotonically increasing counter per process and provides methods to
    simulate internal, send, and receive events according to Lamport's
    algorithm.
    """

    def __init__(self, process_ids: Iterable[int]) -> None:
        """
        Initialise the engine with a collection of process identifiers.

        All counters start at 0. Duplicate identifiers are ignored.
        """
        self._counters: Dict[int, int] = {}
        for pid in process_ids:
            if not isinstance(pid, int):
                raise TypeError(f"Process id must be int, got {type(pid)}")
            if pid < 0:
                raise ValueError(f"Process id must be non‑negative, got {pid}")
            self._counters[pid] = 0

    def _ensure_process(self, pid: int) -> None:
        if pid not in self._counters:
            raise ValueError(f"Unknown process id {pid}")

    def internal_event(self, pid: int) -> int:
        """
        Record an internal event for `pid`.

        Returns the new logical timestamp after increment.
        """
        self._ensure_process(pid)
        self._counters[pid] += 1
        return self._counters[pid]

    def send_event(self, pid: int, target: int) -> Message:
        """
        Record a send event from `pid` to `target`.

        The sender's clock is incremented, and a Message containing the
        new timestamp is returned.
        """
        self._ensure_process(pid)
        self._ensure_process(target)
        self._counters[pid] += 1
        ts = self._counters[pid]
        return Message(sender=pid, receiver=target, timestamp=ts)

    def receive_event(self, pid: int, msg: Message) -> int:
        """
        Record a receive event for `pid` using the supplied `msg`.

        The receiver's clock is set to max(local, msg.timestamp) + 1.
        Returns the new logical timestamp.
        """
        self._ensure_process(pid)
        if msg.receiver != pid:
            raise ValueError(f"Message intended for {msg.receiver}, not {pid}")
        # Apply Lamport rule
        local = self._counters[pid]
        self._counters[pid] = max(local, msg.timestamp) + 1
        return self._counters[pid]

    def get_timestamp(self, pid: int) -> int:
        """Return the current logical timestamp of `pid`."""
        self._ensure_process(pid)
        return self._counters[pid]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._counters})"
