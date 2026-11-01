from __future__ import annotations
from typing import Dict, List, Tuple
from custom_types import RaftNode, NodeId, Message, MessageType, NodeState


class InMemoryNetwork:
    """
    Very small deterministic message bus used by unit tests.
    Each node registers a receive callback; messages are queued and
    delivered in FIFO order when `deliver_all` is called.
    """

    def __init__(self) -> None:
        self.queues: Dict[NodeId, List[Message]] = {}
        self.nodes: Dict[NodeId, RaftNode] = {}

    def register(self, node: RaftNode) -> None:
        self.nodes[node.node_id] = node
        self.queues[node.node_id] = []

    def send(self, msg: Message) -> None:
        # Directly enqueue for the destination node
        self.queues[msg.dst].append(msg)

    def deliver_all(self) -> None:
        # Deliver messages in deterministic order of node ids
        for nid in sorted(self.queues.keys()):
            q = self.queues[nid]
            while q:
                msg = q.pop(0)
                self.nodes[nid].receive(msg)


def run_simulation(step_limit: int = 50) -> Tuple[Dict[NodeId, RaftNode], InMemoryNetwork]:
    """
    Build a 3-node cluster, run deterministic ticks, and return final state.
    """
    net = InMemoryNetwork()
    node_ids = [1, 2, 3]
    nodes: Dict[NodeId, RaftNode] = {}

    for nid in node_ids:
        peers = [pid for pid in node_ids if pid != nid]
        node = RaftNode(
            node_id=nid,
            peer_ids=peers,
            send=net.send,
            election_timeout=5,
            heartbeat_interval=2,
        )
        net.register(node)
        nodes[nid] = node

    # Run the simulation
    for step in range(step_limit):
        # 1. each node ticks
        for node in nodes.values():
            node.tick()
        # 2. deliver all pending messages
        net.deliver_all()
        # 3. check if a stable leader exists (no further elections)
        leaders = [n for n in nodes.values() if n.state == NodeState.LEADER]
        if len(leaders) == 1:
            # Verify that all others are followers and term matches
            leader = leaders[0]
            if all(
                (other.state == NodeState.FOLLOWER and other.current_term == leader.current_term)
                for other in nodes.values()
                if other.node_id != leader.node_id
            ):
                break
    return nodes, net


if __name__ == "__main__":
    # ------------------- Unit Test 1: Single leader election ------------------- #
    nodes, _ = run_simulation()
    leader_nodes = [n for n in nodes.values() if n.state == NodeState.LEADER]
    assert len(leader_nodes) == 1, "Exactly one leader must be elected"
    leader = leader_nodes[0]
    for nid, node in nodes.items():
        if nid != leader.node_id:
            assert node.state == NodeState.FOLLOWER, f"Node {nid} should be follower"
            assert node.current_term == leader.current_term, "Terms must be equal across cluster"

    # ------------------- Unit Test 2: Higher term forces step down ------------------- #
    # Create two nodes, force node 1 to become leader, then send higher term AppendEntries to it.
    net2 = InMemoryNetwork()
    node_a = RaftNode(node_id=1, peer_ids=[2], send=net2.send, election_timeout=3, heartbeat_interval=2)
    node_b = RaftNode(node_id=2, peer_ids=[1], send=net2.send, election_timeout=3, heartbeat_interval=2)
    net2.register(node_a)
    net2.register(node_b)

    # Simulate enough ticks for node_a to become leader
    for _ in range(4):
        node_a.tick()
        node_b.tick()
        net2.deliver_all()

    assert node_a.state == NodeState.LEADER, "Node A should have become leader"
    assert node_b.state == NodeState.FOLLOWER, "Node B should be follower"

    # Now inject a fake AppendEntries with higher term from node_b
    higher_term_msg = Message(
        src=2,
        dst=1,
        term=node_b.current_term + 1,
        type=MessageType.APPEND_ENTRIES,
        payload=None,
    )
    node_a.receive(higher_term_msg)

    assert node_a.state == NodeState.FOLLOWER, "Leader must step down on higher term"
    assert node_a.current_term == node_b.current_term + 1, "Term must be updated to higher term"

    # ------------------- Unit Test 3: Split vote resolves after timeout ------------------- #
    # Three nodes where two candidates start election simultaneously.
    net3 = InMemoryNetwork()
    ids = [10, 20, 30]
    nodes3: Dict[NodeId, RaftNode] = {}
    for nid in ids:
        peers = [pid for pid in ids if pid != nid]
        node = RaftNode(
            node_id=nid,
            peer_ids=peers,
            send=net3.send,
            election_timeout=3,   # all same timeout to cause split vote
            heartbeat_interval=2,
        )
        net3.register(node)
        nodes3[nid] = node

    # Force all nodes to start election at the same tick
    for node in nodes3.values():
        node.timer = node.election_timeout  # trigger election on next tick

    # Run a few steps; the split vote should cause another election round
    for _ in range(6):
        for node in nodes3.values():
            node.tick()
        net3.deliver_all()

    leaders = [n for n in nodes3.values() if n.state == NodeState.LEADER]
    assert len(leaders) == 1, "After split vote, a single leader must eventually be elected"
    final_leader = leaders[0]
    for nid, node in nodes3.items():
        if nid != final_leader.node_id:
            assert node.state == NodeState.FOLLOWER, "Non-leader nodes must be followers"
            assert node.current_term == final_leader.current_term, "Terms must converge"

    print("All Raft leader election unit tests passed.")
