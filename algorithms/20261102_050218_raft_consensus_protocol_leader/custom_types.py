from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Dict, Any

NodeId = int
Term = int
LogIndex = int


class NodeState(Enum):
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


@dataclass(frozen=True)
class Message:
    src: NodeId
    dst: NodeId
    term: Term


@dataclass(frozen=True)
class RequestVote(Message):
    last_log_index: LogIndex
    last_log_term: Term


@dataclass(frozen=True)
class RequestVoteResponse(Message):
    vote_granted: bool


@dataclass(frozen=True)
class AppendEntries(Message):
    leader_id: NodeId
    prev_log_index: LogIndex
    prev_log_term: Term
    entries: List[Any]  # log entries are abstracted; not used in election engine
    leader_commit: LogIndex


@dataclass(frozen=True)
class AppendEntriesResponse(Message):
    success: bool
    match_index: LogIndex
