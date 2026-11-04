from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True, order=True)
class Event:
    """
    Represents a single event in a distributed system.
    Attributes:
        process_id: Identifier of the process where the event occurred.
        local_counter: The local sequence number within the process.
        timestamp: The Lamport timestamp assigned to the event.
    """
    process_id: int
    local_counter: int
    timestamp: int


class LamportClock:
    """
    A classic Lamport logical clock.
    Guarantees:
        * Monotonic increase on local events.
        * On receipt of a message with timestamp t, the clock becomes max(t, local) + 1.
    """

    def __init__(self, pid: int) -> None:
        self.pid: int = pid
        self.counter: int = 0

    def tick(self) -> int:
        """
        Register a local event and advance the clock.
        Returns the new timestamp.
        """
        self.counter += 1
        return self.counter

    def update(self, received_timestamp: int) -> int:
        """
        Incorporate a timestamp received from another process.
        Returns the new timestamp after the update.
        """
        self.counter = max(self.counter, received_timestamp) + 1
        return self.counter

    def get_time(self) -> int:
        """Current logical time without advancing."""
        return self.counter

    def event(self) -> Event:
        """
        Convenience: create an Event instance for the current local event.
        The local_counter equals the timestamp because each tick increments by one.
        """
        ts = self.tick()
        return Event(process_id=self.pid, local_counter=ts, timestamp=ts)


def total_order(event_a: Event, event_b: Event) -> int:
    """
    Total order used by Lamport timestamps.
    First compare timestamps, then break ties with process identifiers.
    Returns:
        -1 if a < b, 1 if a > b, 0 if identical (same process and counter).
    """
    if event_a.timestamp < event_b.timestamp:
        return -1
    if event_a.timestamp > event_b.timestamp:
        return 1
    # timestamps equal – break tie deterministically by pid
    if event_a.process_id < event_b.process_id:
        return -1
    if event_a.process_id > event_b.process_id:
        return 1
    # identical events (should not happen in a well‑formed history)
    return 0


def causally_related(event_a: Event, event_b: Event, history: List[Tuple[Event, List[int]]]) -> bool:
    """
    Determine whether event_a causally precedes event_b using the happened‑before relation.
    The `history` argument is a list of (event, message_targets) tuples where
    `message_targets` contains the timestamps of events that directly receive a message
    from `event`. This is a minimal representation sufficient for the unit tests.

    Returns True iff there exists a chain of messages from a to b.
    """
    # Build adjacency list from timestamps to events
    adj = {}
    for ev, targets in history:
        adj.setdefault(ev.timestamp, []).extend(targets)

    # Depth‑first search from a.timestamp to see if we can reach b.timestamp
    visited = set()
    stack = [event_a.timestamp]
    while stack:
        cur = stack.pop()
        if cur == event_b.timestamp:
            return True
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(adj.get(cur, []))
    return False
