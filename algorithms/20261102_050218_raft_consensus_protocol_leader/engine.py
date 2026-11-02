from __future__ import annotations
from typing import List, Dict, Set
import itertools

from custom_types import (
    NodeId,
    Term,
    LogIndex,
    NodeState,
    Message,
    RequestVote,
    RequestVoteResponse,
    AppendEntries,
    AppendEntriesResponse,
)


class RaftNode:
    """
    Minimal Raft node implementing only the leader election state machine.
    All log‑related fields are stubbed because they are irrelevant for the
    election proof‑of‑concept.
    """

    def __init__(self, node_id: NodeId, peers: List[NodeId], cluster: "RaftCluster"):
        self.id: NodeId = node_id
        self.peers: List[NodeId] = peers  # all other node ids
        self.cluster: RaftCluster = cluster

        # Persistent state on all servers
        self.current_term: Term = 0
        self.voted_for: Optional[NodeId] = None

        # Volatile state
        self.state: NodeState = NodeState.FOLLOWER
        self.votes_received: Set[NodeId] = set()

        # Timing (deterministic, no real time)
        self.election_timeout: int = self._base_election_timeout()
        self.heartbeat_interval: int = 2
        self._timer: int = 0  # counts ticks since last reset

        # For testability: a flag to disable this node (simulating crash)
        self.active: bool = True

    # --------------------------------------------------------------------- #
    # Timing helpers
    # --------------------------------------------------------------------- #
    def _base_election_timeout(self) -> int:
        """
        Deterministic election timeout based on node id.
        Guarantees that not all nodes timeout on the same tick.
        """
        base = 5
        return base + (self.id % 3)  # values: 5,6,7

    def _reset_timer(self) -> None:
        self._timer = 0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def tick(self) -> None:
        """Advance one logical tick; may trigger state transitions."""
        if not self.active:
            return

        self._timer += 1

        if self.state == NodeState.LEADER:
            if self._timer >= self.heartbeat_interval:
                self._send_heartbeats()
                self._reset_timer()
        else:
            if self._timer >= self.election_timeout:
                self._start_election()
                self._reset_timer()

    def receive(self, msg: Message) -> None:
        """Dispatch incoming messages to the appropriate handler."""
        if not self.active:
            return

        if isinstance(msg, RequestVote):
            self._handle_request_vote(msg)
        elif isinstance(msg, RequestVoteResponse):
            self._handle_request_vote_response(msg)
        elif isinstance(msg, AppendEntries):
            self._handle_append_entries(msg)
        elif isinstance(msg, AppendEntriesResponse):
            # In this minimal engine we ignore AppendEntriesResponse
            pass

    # --------------------------------------------------------------------- #
    # Message handlers
    # --------------------------------------------------------------------- #
    def _handle_request_vote(self, msg: RequestVote) -> None:
        grant = False
        if msg.term < self.current_term:
            grant = False
        else:
            # If term is higher, step down first
            if msg.term > self.current_term:
                self._become_follower(msg.term)

            if (self.voted_for is None or self.voted_for == msg.src):
                # In a full Raft implementation we would also compare logs.
                grant = True
                self.voted_for = msg.src
                self._reset_timer()  # grant resets election timer

        response = RequestVoteResponse(
            src=self.id,
            dst=msg.src,
            term=self.current_term,
            vote_granted=grant,
        )
        self.cluster.send_message(response)

    def _handle_request_vote_response(self, msg: RequestVoteResponse) -> None:
        if self.state != NodeState.CANDIDATE:
            return
        if msg.term > self.current_term:
            self._become_follower(msg.term)
            return
        if msg.term < self.current_term:
            return  # stale response

        if msg.vote_granted:
            self.votes_received.add(msg.src)
            if self._has_majority():
                self._become_leader()

    def _handle_append_entries(self, msg: AppendEntries) -> None:
        if msg.term < self.current_term:
            # Reply false; leader is stale
            response = AppendEntriesResponse(
                src=self.id,
                dst=msg.src,
                term=self.current_term,
                success=False,
                match_index=0,
            )
            self.cluster.send_message(response)
            return

        # If term is newer, step down
        if msg.term > self.current_term:
            self._become_follower(msg.term)

        # Reset election timer because we received valid leader heartbeat
        self._reset_timer()
        # In full Raft we would also append entries; omitted here.

        response = AppendEntriesResponse(
            src=self.id,
            dst=msg.src,
            term=self.current_term,
            success=True,
            match_index=0,
        )
        self.cluster.send_message(response)

    # --------------------------------------------------------------------- #
    # State transition helpers
    # --------------------------------------------------------------------- #
    def _start_election(self) -> None:
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.id
        self.votes_received = {self.id}
        self._reset_timer()

        # Broadcast RequestVote to all peers
        for peer in self.peers:
            rv = RequestVote(
                src=self.id,
                dst=peer,
                term=self.current_term,
                last_log_index=0,
                last_log_term=0,
            )
            self.cluster.send_message(rv)

    def _become_follower(self, term: Term) -> None:
        self.state = NodeState.FOLLOWER
        self.current_term = term
        self.voted_for = None
        self.votes_received.clear()
        self._reset_timer()

    def _become_leader(self) -> None:
        self.state = NodeState.LEADER
        self.voted_for = None
        self.votes_received.clear()
        self._reset_timer()
        self._send_heartbeats()  # immediate heartbeat upon election

    def _send_heartbeats(self) -> None:
        for peer in self.peers:
            hb = AppendEntries(
                src=self.id,
                dst=peer,
                term=self.current_term,
                leader_id=self.id,
                prev_log_index=0,
                prev_log_term=0,
                entries=[],
                leader_commit=0,
            )
            self.cluster.send_message(hb)

    def _has_majority(self) -> bool:
        total = len(self.peers) + 1  # include self
        return len(self.votes_received) > total // 2


class RaftCluster:
    """
    Simple in‑process cluster that routes messages between RaftNode instances.
    No real networking; all calls are synchronous and deterministic.
    """

    def __init__(self, node_ids: List[NodeId]):
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Node IDs must be unique")
        self.nodes: Dict[NodeId, RaftNode] = {}
        for nid in node_ids:
            peers = [pid for pid in node_ids if pid != nid]
            self.nodes[nid] = RaftNode(node_id=nid, peers=peers, cluster=self)

    def send_message(self, msg: Message) -> None:
        """Synchronously deliver a message to its destination."""
        dst_node = self.nodes.get(msg.dst)
        if dst_node is None:
            raise ValueError(f"Destination node {msg.dst} does not exist")
        dst_node.receive(msg)

    def tick_all(self) -> None:
        """Advance one logical tick on every active node."""
        for node in self.nodes.values():
            node.tick()

    def get_leader(self) -> Optional[RaftNode]:
        """Return the current leader if exactly one exists, else None."""
        leaders = [n for n in self.nodes.values() if n.state == NodeState.LEADER and n.active]
        if len(leaders) == 1:
            return leaders[0]
        return None

    def deactivate_node(self, node_id: NodeId) -> None:
        """Simulate a crash by disabling a node."""
        node = self.nodes.get(node_id)
        if node:
            node.active = False

    def activate_node(self, node_id: NodeId) -> None:
        """Reactivate a previously deactivated node."""
        node = self.nodes.get(node_id)
        if node:
            node.active = True
            node._reset_timer()
