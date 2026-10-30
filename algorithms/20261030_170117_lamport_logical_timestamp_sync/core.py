from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, order=True)
class Timestamp:
    """
    Lamport timestamp consisting of a logical counter and a process identifier.
    Ordering is first by counter, then by pid to break ties deterministically.
    """
    counter: int
    pid: int


@dataclass(frozen=True)
class Event:
    """
    An event occurring in a process, identified by a name and its Lamport timestamp.
    """
    name: str
    timestamp: Timestamp


class LamportClock:
    """
    Simple Lamport logical clock bound to a single process.
    """

    def __init__(self, pid: int) -> None:
        self.pid: int = pid
        self.counter: int = 0

    def tick(self) -> int:
        """
        Advance the clock for a local event and return the new counter.
        """
        self.counter += 1
        return self.counter

    def update(self, remote_counter: int) -> int:
        """
        Incorporate a remote timestamp (from a message) and return the new counter.
        """
        self.counter = max(self.counter, remote_counter) + 1
        return self.counter

    def timestamp(self) -> Timestamp:
        """
        Produce a Timestamp object reflecting the current state.
        """
        return Timestamp(self.counter, self.pid)


class Process:
    """
    Represents a logical process that can generate local events,
    send messages, and receive messages while maintaining a Lamport clock.
    """

    def __init__(self, pid: int) -> None:
        self.pid: int = pid
        self.clock: LamportClock = LamportClock(pid)
        self._log: List[Event] = []

    def _record(self, name: str) -> Timestamp:
        ts = self.clock.timestamp()
        self._log.append(Event(name, ts))
        return ts

    def local_event(self, name: str) -> Timestamp:
        """
        Generate a purely local event.
        """
        self.clock.tick()
        return self._record(name)

    def send(self, name: str) -> Timestamp:
        """
        Model sending a message; for Lamport clocks this is equivalent to a local event.
        """
        return self.local_event(name)

    def receive(self, remote_ts: Timestamp, name: str) -> Timestamp:
        """
        Model receiving a message with a remote timestamp.
        """
        self.clock.update(remote_ts.counter)
        return self._record(name)

    def get_log(self) -> List[Event]:
        """
        Return a copy of the process's event log.
        """
        return list(self._log)


def merge_logs(*processes: Process) -> List[Event]:
    """
    Produce a globally ordered list of events from the supplied processes.
    Ordering follows Lamport's total order: first by counter, then by pid.
    """
    all_events: List[Event] = []
    for proc in processes:
        all_events.extend(proc.get_log())
    return sorted(all_events, key=lambda e: (e.timestamp.counter, e.timestamp.pid))
