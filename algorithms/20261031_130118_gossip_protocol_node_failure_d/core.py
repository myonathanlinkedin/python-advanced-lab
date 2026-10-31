import random
import time
from enum import Enum, auto
from typing import Dict, List, Callable, Tuple


class NodeStatus(Enum):
    ALIVE = auto()
    SUSPECTED = auto()
    FAILED = auto()


class Node:
    __slots__ = ("node_id", "heartbeat", "last_update", "status")

    def __init__(self, node_id: int, init_time: float) -> None:
        self.node_id: int = node_id
        self.heartbeat: int = 0
        self.last_update: float = init_time
        self.status: NodeStatus = NodeStatus.ALIVE

    def increment_heartbeat(self) -> None:
        self.heartbeat += 1
        self.last_update = time.time()


class GossipFailureDetector:
    """
    Simulated gossip based failure detector.
    Deterministic behaviour is achieved by providing a fixed seed to the internal RNG.
    """

    def __init__(
        self,
        node_ids: List[int],
        gossip_interval: float = 1.0,
        failure_threshold: int = 3,
        clock: Callable[[], float] = time.time,
        rng_seed: int = 42,
    ) -> None:
        if not node_ids:
            raise ValueError("node_ids must contain at least one identifier")
        self._clock = clock
        self.gossip_interval = gossip_interval
        self.failure_threshold = failure_threshold
        self._rng = random.Random(rng_seed)

        init = self._clock()
        self.nodes: Dict[int, Node] = {nid: Node(nid, init) for nid in node_ids}
        # For simulation we keep a set of nodes that are allowed to gossip.
        self.active_node_ids: set[int] = set(node_ids)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def tick(self) -> None:
        """
        Simulate one gossip round. Each active node selects a random peer
        (including possibly itself) and exchanges heartbeat counters.
        """
        now = self._clock()
        # Step 1: each node increments its own heartbeat (simulating local health)
        for nid in self.active_node_ids:
            node = self.nodes[nid]
            node.increment_heartbeat()

        # Step 2: gossip exchange
        for nid in self.active_node_ids:
            peer_id = self._select_peer(nid)
            self._exchange(nid, peer_id)

        # Step 3: evaluate failures based on timestamps
        self._evaluate_statuses(now)

    def get_status(self, node_id: int) -> NodeStatus:
        return self.nodes[node_id].status

    def get_all_statuses(self) -> Dict[int, NodeStatus]:
        return {nid: n.status for nid, n in self.nodes.items()}

    def deactivate_node(self, node_id: int) -> None:
        """Prevent a node from participating in future gossip rounds."""
        self.active_node_ids.discard(node_id)

    def activate_node(self, node_id: int) -> None:
        """Re‑enable a node for gossip."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} unknown")
        self.active_node_ids.add(node_id)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _select_peer(self, source_id: int) -> int:
        """Select a random peer from the current node set (including self)."""
        candidates = list(self.nodes.keys())
        # deterministic choice
        return self._rng.choice(candidates)

    def _exchange(self, src_id: int, dst_id: int) -> None:
        """Exchange heartbeat counters between two nodes."""
        src = self.nodes[src_id]
        dst = self.nodes[dst_id]

        # The node with the higher heartbeat wins; the other adopts it.
        if src.heartbeat > dst.heartbeat:
            dst.heartbeat = src.heartbeat
            dst.last_update = self._clock()
        elif dst.heartbeat > src.heartbeat:
            src.heartbeat = dst.heartbeat
            src.last_update = self._clock()
        # If equal, nothing changes.

    def _evaluate_statuses(self, now: float) -> None:
        """Mark nodes as SUSPECTED or FAILED based on elapsed time."""
        timeout = self.failure_threshold * self.gossip_interval
        for node in self.nodes.values():
            elapsed = now - node.last_update
            if node.status == NodeStatus.ALIVE and elapsed > timeout:
                node.status = NodeStatus.SUSPECTED
            elif node.status == NodeStatus.SUSPECTED and elapsed > 2 * timeout:
                node.status = NodeStatus.FAILED
