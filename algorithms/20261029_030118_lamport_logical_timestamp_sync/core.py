from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Message:
    """Immutable message carrying a Lamport timestamp."""
    sender_id: str
    timestamp: int
    payload: str = ""  # optional payload for extensibility


class LamportClock:
    """Lamport logical clock for a single process."""

    def __init__(self, start: int = 0) -> None:
        if start < 0:
            raise ValueError("Lamport clock cannot start with a negative value.")
        self._counter: int = start

    @property
    def time(self) -> int:
        """Current logical time."""
        return self._counter

    def tick(self) -> int:
        """Advance clock for an internal event or before sending."""
        self._counter += 1
        return self._counter

    def update(self, received: int) -> int:
        """
        Update clock on receipt of a message.
        The new time is max(local, received) + 1.
        """
        if received < 0:
            raise ValueError("Received timestamp must be non‑negative.")
        self._counter = max(self._counter, received) + 1
        return self._counter

    def __repr__(self) -> str:
        return f"LamportClock(time={self._counter})"


class Process:
    """
    Represents a process participating in Lamport timestamp synchronization.
    It maintains its own logical clock and a log of events for verification.
    """

    def __init__(self, pid: str, start: int = 0) -> None:
        self.pid: str = pid
        self.clock: LamportClock = LamportClock(start)
        # Event log entries: (event_type, timestamp)
        self.log: List[Tuple[str, int]] = []

    def internal_event(self) -> int:
        """Perform an internal event; increments the clock."""
        ts = self.clock.tick()
        self.log.append(("internal", ts))
        return ts

    def send(self, payload: str = "") -> Message:
        """
        Prepare a message for sending.
        The clock is ticked before the timestamp is attached.
        """
        ts = self.clock.tick()
        self.log.append(("send", ts))
        return Message(sender_id=self.pid, timestamp=ts, payload=payload)

    def receive(self, msg: Message) -> int:
        """
        Process a received message.
        Updates the logical clock according to Lamport's rule.
        """
        ts = self.clock.update(msg.timestamp)
        self.log.append(("receive", ts))
        return ts

    def __repr__(self) -> str:
        return f"Process(pid={self.pid!r}, clock={self.clock})"


class LamportEngine:
    """
    Engine coordinating a set of processes.
    It provides deterministic message delivery without real networking.
    """

    def __init__(self) -> None:
        self.processes: Dict[str, Process] = {}

    def add_process(self, pid: str, start: int = 0) -> Process:
        """Register a new process; raises if pid already exists."""
        if pid in self.processes:
            raise KeyError(f"Process with id {pid!r} already exists.")
        proc = Process(pid, start)
        self.processes[pid] = proc
        return proc

    def get_process(self, pid: str) -> Process:
        """Retrieve a process by identifier."""
        try:
            return self.processes[pid]
        except KeyError as exc:
            raise KeyError(f"Process {pid!r} not found.") from exc

    def send_message(self, sender_id: str, receiver_id: str, payload: str = "") -> Message:
        """
        Simulate sending a message from sender to receiver.
        Returns the Message object for inspection.
        """
        sender = self.get_process(sender_id)
        receiver = self.get_process(receiver_id)
        msg = sender.send(payload)
        # Deterministic delivery: immediate receipt.
        receiver.receive(msg)
        return msg

    def broadcast(self, sender_id: str, payload: str = "") -> List[Message]:
        """
        Send a message from sender to all other processes.
        Returns the list of delivered messages.
        """
        msgs: List[Message] = []
        for pid in self.processes:
            if pid != sender_id:
                msgs.append(self.send_message(sender_id, pid, payload))
        return msgs

    def __repr__(self) -> str:
        return f"LamportEngine(processes={list(self.processes)})"


__all__ = ["Message", "LamportClock", "Process", "LamportEngine"]
