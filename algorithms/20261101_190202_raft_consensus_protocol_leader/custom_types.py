from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


class NodeState(Enum):
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


class MessageType(Enum):
    REQUEST_VOTE = auto()
    REQUEST_VOTE_RESPONSE = auto()
    APPEND_ENTRIES = auto()          # Used as heartbeat in this simplified engine
    APPEND_ENTRIES_RESPONSE = auto()


NodeId = int


@dataclass(frozen=True)
class Message:
    src: NodeId
    dst: NodeId
    term: int
    type: MessageType
    payload: Any = None
