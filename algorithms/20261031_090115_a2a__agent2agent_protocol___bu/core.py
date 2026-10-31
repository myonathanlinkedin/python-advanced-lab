"""
Core implementation of a minimal Agent‑to‑Agent (A2A) protocol using
vector clocks to guarantee causal ordering of messages.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple


@dataclass(frozen=True)
class VectorClock:
    """Immutable vector clock."""
    clock: Tuple[Tuple[str, int], ...]  # sorted tuple of (agent_id, counter)

    @staticmethod
    def empty() -> VectorClock:
        return VectorClock(())

    @staticmethod
    def from_dict(d: Dict[str, int]) -> VectorClock:
        # Store entries sorted to guarantee hashability & equality
        items = tuple(sorted((k, v) for k, v in d.items() if v > 0))
        return VectorClock(items)

    def to_dict(self) -> Dict[str, int]:
        return dict(self.clock)

    def increment(self, agent_id: str) -> VectorClock:
        d = self.to_dict()
        d[agent_id] = d.get(agent_id, 0) + 1
        return VectorClock.from_dict(d)

    def merge(self, other: VectorClock) -> VectorClock:
        d1 = self.to_dict()
        d2 = other.to_dict()
        merged = {k: max(d1.get(k, 0), d2.get(k, 0)) for k in set(d1) | set(d2)}
        return VectorClock.from_dict(merged)

    def __getitem__(self, agent_id: str) -> int:
        return dict(self.clock).get(agent_id, 0)

    def __le__(self, other: VectorClock) -> bool:
        """Component‑wise less‑or‑equal."""
        d1 = self.to_dict()
        d2 = other.to_dict()
        for k in d1:
            if d1[k] > d2.get(k, 0):
                return False
        return True

    def __repr__(self) -> str:
        return f"VectorClock({self.to_dict()})"


@dataclass
class Message:
    """A message exchanged between agents."""
    sender: str
    receiver: str
    payload: Any
    timestamp: VectorClock


class Agent:
    """Represents an autonomous participant in the A2A protocol."""

    def __init__(self, agent_id: str):
        self.id: str = agent_id
        self._clock: VectorClock = VectorClock.empty()
        self.inbox: List[Message] = []

    @property
    def clock(self) -> VectorClock:
        return self._clock

    def _increment_clock(self) -> None:
        self._clock = self._clock.increment(self.id)

    def send(self, receiver: str, payload: Any) -> Message:
        """
        Produce a message destined for `receiver`. The agent's clock is
        incremented before the message is created.
        """
        self._increment_clock()
        msg = Message(
            sender=self.id,
            receiver=receiver,
            payload=payload,
            timestamp=self._clock,
        )
        return msg

    def can_deliver(self, msg: Message) -> bool:
        """
        Causal delivery condition:
        - For all agents k ≠ sender: local_clock[k] ≥ msg.timestamp[k]
        - For sender: local_clock[sender] + 1 == msg.timestamp[sender]
        """
        local = self._clock
        ts = msg.timestamp
        # Sender condition
        if local[msg.sender] + 1 != ts[msg.sender]:
            return False
        # Other agents condition
        for agent, count in ts.to_dict().items():
            if agent == msg.sender:
                continue
            if local[agent] < count:
                return False
        return True

    def receive(self, msg: Message) -> None:
        """
        Attempt to deliver `msg`. If causal condition fails, raise ValueError.
        On success, merge clocks and store the message in the inbox.
        """
        if not self.can_deliver(msg):
            raise ValueError(f"Causal violation when delivering {msg}")
        # Merge remote clock (excluding the sender's increment already accounted)
        merged = self._clock.merge(msg.timestamp)
        # Increment own entry to reflect receipt
        self._clock = merged.increment(self.id)
        self.inbox.append(msg)

    def __repr__(self) -> str:
        return f"Agent(id={self.id}, clock={self._clock})"


class NetworkSimulator:
    """
    Deterministic in‑memory network that stores pending messages.
    The `deliver_all` method attempts to deliver messages respecting
    causal ordering; undeliverable messages remain queued.
    """

    def __init__(self, agents: List[Agent]):
        self.agents: Dict[str, Agent] = {a.id: a for a in agents}
        self.pending: List[Message] = []

    def send(self, msg: Message) -> None:
        if msg.receiver not in self.agents:
            raise KeyError(f"Unknown receiver {msg.receiver}")
        self.pending.append(msg)

    def deliver_all(self) -> None:
        """
        Repeatedly scan pending messages and deliver any that satisfy
        the causal condition. Stops when a full pass yields no deliveries.
        """
        delivered = True
        while delivered:
            delivered = False
            remaining: List[Message] = []
            for msg in self.pending:
                receiver = self.agents[msg.receiver]
                if receiver.can_deliver(msg):
                    receiver.receive(msg)
                    delivered = True
                else:
                    remaining.append(msg)
            self.pending = remaining

    def __repr__(self) -> str:
        return f"NetworkSimulator(pending={len(self.pending)})"
