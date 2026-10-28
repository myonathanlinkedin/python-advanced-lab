from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


class LamportClock:
    """Monotonic logical clock following Lamport's algorithm."""

    def __init__(self) -> None:
        self.counter: int = 0

    def tick(self) -> int:
        """Internal event: increment and return new timestamp."""
        self.counter += 1
        return self.counter

    def send(self) -> int:
        """Send event: increment before sending."""
        return self.tick()

    def receive(self, remote_ts: int) -> int:
        """Receive event: set to max(local, remote) + 1."""
        self.counter = max(self.counter, remote_ts) + 1
        return self.counter

    def value(self) -> int:
        return self.counter


@dataclass(frozen=True)
class Message:
    """Immutable message carrying a Lamport timestamp."""
    sender_id: str
    receiver_id: str
    timestamp: int


@dataclass
class Process:
    """A logical process equipped with a Lamport clock."""
    pid: str
    clock: LamportClock = field(default_factory=LamportClock)
    # event log entries: (event_type, timestamp, optional partner)
    log: List[Tuple[str, int, str | None]] = field(default_factory=list)

    def internal_event(self) -> int:
        ts = self.clock.tick()
        self.log.append(("internal", ts, None))
        return ts

    def send(self, receiver_id: str) -> Message:
        ts = self.clock.send()
        self.log.append(("send", ts, receiver_id))
        return Message(sender_id=self.pid, receiver_id=receiver_id, timestamp=ts)

    def receive(self, msg: Message) -> int:
        if msg.receiver_id != self.pid:
            raise ValueError(f"Message intended for {msg.receiver_id}, received by {self.pid}")
        ts = self.clock.receive(msg.timestamp)
        self.log.append(("receive", ts, msg.sender_id))
        return ts

    def get_log(self) -> List[Tuple[str, int, str | None]]:
        return list(self.log)


class LamportEngine:
    """Engine coordinating multiple processes and message delivery."""

    def __init__(self) -> None:
        self.processes: Dict[str, Process] = {}
        self.delivered_messages: List[Message] = []

    def add_process(self, pid: str) -> Process:
        if pid in self.processes:
            raise ValueError(f"Process {pid} already exists.")
        proc = Process(pid=pid)
        self.processes[pid] = proc
        return proc

    def get_process(self, pid: str) -> Process:
        return self.processes[pid]

    def send_message(self, sender_id: str, receiver_id: str) -> Message:
        sender = self.get_process(sender_id)
        if receiver_id not in self.processes:
            raise ValueError(f"Receiver {receiver_id} not registered.")
        msg = sender.send(receiver_id)
        self.deliver_message(msg)
        return msg

    def deliver_message(self, msg: Message) -> None:
        receiver = self.get_process(msg.receiver_id)
        receiver.receive(msg)
        self.delivered_messages.append(msg)

    def verify_causality(self) -> bool:
        """
        Verify that for every delivered message, the sender's timestamp
        is strictly less than the receiver's timestamp after processing.
        """
        for msg in self.delivered_messages:
            sender_ts = self.processes[msg.sender_id].clock.value()
            receiver_ts = self.processes[msg.receiver_id].clock.value()
            if not (msg.timestamp < receiver_ts):
                return False
        return True

    def snapshot(self) -> Dict[str, int]:
        """Current logical time of each process."""
        return {pid: proc.clock.value() for pid, proc in self.processes.items()}
