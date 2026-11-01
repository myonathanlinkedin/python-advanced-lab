from __future__ import annotations
from typing import Callable, Dict, List, Set
from custom_types import NodeState, Message, MessageType, NodeId


class RaftNode:
    """
    Simplified Raft state machine focusing on leader election.
    All time is expressed in discrete ticks.
    """

    def __init__(
        self,
        node_id: NodeId,
        peer_ids: List[NodeId],
        send: Callable[[Message], None],
        election_timeout: int = 5,
        heartbeat_interval: int = 2,
    ) -> None:
        self.node_id: NodeId = node_id
        self.peers: List[NodeId] = peer_ids
        self.send = send

        self.state: NodeState = NodeState.FOLLOWER
        self.current_term: int = 0
        self.voted_for: Optional[NodeId] = None

        # election/heartbeat timers (ticks since last reset)
        self.election_timeout: int = election_timeout
        self.heartbeat_interval: int = heartbeat_interval
        self.timer: int = 0

        # candidate state
        self.votes_received: Set[NodeId] = set()

    # --------------------------------------------------------------------- #
    # Timer handling
    # --------------------------------------------------------------------- #
    def reset_election_timer(self) -> None:
        self.timer = 0

    def tick(self) -> None:
        """
        Advance one logical tick.
        """
        self.timer += 1
        if self.state == NodeState.LEADER:
            if self.timer >= self.heartbeat_interval:
                self._send_heartbeats()
                self.timer = 0
        else:
            if self.timer >= self.election_timeout:
                self._start_election()
                self.timer = 0

    # --------------------------------------------------------------------- #
    # Message handling entry point
    # --------------------------------------------------------------------- #
    def receive(self, msg: Message) -> None:
        if msg.term > self.current_term:
            self._become_follower(msg.term)
        if msg.type == MessageType.REQUEST_VOTE:
            self._on_request_vote(msg)
        elif msg.type == MessageType.REQUEST_VOTE_RESPONSE:
            self._on_request_vote_response(msg)
        elif msg.type == MessageType.APPEND_ENTRIES:
            self._on_append_entries(msg)
        # APPEND_ENTRIES_RESPONSE is ignored in this simplified engine

    # --------------------------------------------------------------------- #
    # State transitions
    # --------------------------------------------------------------------- #
    def _become_follower(self, term: int) -> None:
        self.state = NodeState.FOLLOWER
        self.current_term = term
        self.voted_for = None
        self.votes_received.clear()
        self.reset_election_timer()

    def _become_candidate(self) -> None:
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self.reset_election_timer()
        self._broadcast_request_vote()

    def _become_leader(self) -> None:
        self.state = NodeState.LEADER
        self.voted_for = None
        self.votes_received.clear()
        self.timer = 0
        self._send_heartbeats()

    # --------------------------------------------------------------------- #
    # Election logic
    # --------------------------------------------------------------------- #
    def _start_election(self) -> None:
        self._become_candidate()

    def _broadcast_request_vote(self) -> None:
        for peer in self.peers:
            msg = Message(
                src=self.node_id,
                dst=peer,
                term=self.current_term,
                type=MessageType.REQUEST_VOTE,
                payload=None,  # log info omitted for brevity
            )
            self.send(msg)

    def _on_request_vote(self, msg: Message) -> None:
        grant = False
        if msg.term < self.current_term:
            grant = False
        else:
            if (self.voted_for is None or self.voted_for == msg.src):
                # Log up-to-date check omitted; always true in this simplified model
                grant = True
                self.voted_for = msg.src
                self.reset_election_timer()
        response = Message(
            src=self.node_id,
            dst=msg.src,
            term=self.current_term,
            type=MessageType.REQUEST_VOTE_RESPONSE,
            payload=grant,
        )
        self.send(response)

    def _on_request_vote_response(self, msg: Message) -> None:
        if self.state != NodeState.CANDIDATE:
            return
        if msg.term != self.current_term:
            return
        if msg.payload:  # grant == True
            self.votes_received.add(msg.src)
            if self._has_majority():
                self._become_leader()

    def _has_majority(self) -> bool:
        total = len(self.peers) + 1  # include self
        return len(self.votes_received) > total // 2

    # --------------------------------------------------------------------- #
    # Heartbeat (AppendEntries) handling
    # --------------------------------------------------------------------- #
    def _send_heartbeats(self) -> None:
        for peer in self.peers:
            msg = Message(
                src=self.node_id,
                dst=peer,
                term=self.current_term,
                type=MessageType.APPEND_ENTRIES,
                payload=None,  # empty heartbeat
            )
            self.send(msg)

    def _on_append_entries(self, msg: Message) -> None:
        # If term is older, ignore (in real Raft we would reply false)
        if msg.term < self.current_term:
            return
        # Accept leader, become follower if not already
        if self.state != NodeState.FOLLOWER or self.current_term != msg.term:
            self._become_follower(msg.term)
        self.reset_election_timer()
        # No response needed for heartbeat in this simplified engine
