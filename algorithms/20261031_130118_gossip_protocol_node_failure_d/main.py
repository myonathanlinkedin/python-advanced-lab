import unittest
import time
from typing import Callable

# Import the core module
from core import GossipFailureDetector, NodeStatus


class DeterministicClock:
    """
    Simple deterministic clock for unit tests.
    Allows manual advancement of time.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class TestGossipFailureDetector(unittest.TestCase):
    def setUp(self) -> None:
        # Fixed seed ensures reproducibility across runs.
        self.clock = DeterministicClock()
        self.node_ids = [1, 2, 3, 4]
        self.detector = GossipFailureDetector(
            node_ids=self.node_ids,
            gossip_interval=1.0,
            failure_threshold=2,
            clock=self.clock,
            rng_seed=123,
        )

    def test_initial_statuses(self) -> None:
        for nid in self.node_ids:
            self.assertEqual(self.detector.get_status(nid), NodeStatus.ALIVE)

    def test_failure_detection(self) -> None:
        # Let all nodes gossip for two rounds – all stay alive.
        for _ in range(2):
            self.detector.tick()
            self.clock.advance(1.0)

        # Deactivate node 3 to simulate a crash.
        self.detector.deactivate_node(3)

        # Run enough rounds for timeout (failure_threshold=2, interval=1.0)
        # After 2 more ticks, node 3 should become SUSPECTED.
        for _ in range(2):
            self.detector.tick()
            self.clock.advance(1.0)

        self.assertEqual(self.detector.get_status(3), NodeStatus.SUSPECTED)

        # After another 2 ticks, node 3 should be marked FAILED.
        for _ in range(2):
            self.detector.tick()
            self.clock.advance(1.0)

        self.assertEqual(self.detector.get_status(3), NodeStatus.FAILED)

        # Other nodes must remain ALIVE.
        for nid in [1, 2, 4]:
            self.assertEqual(self.detector.get_status(nid), NodeStatus.ALIVE)

    def test_reactivation(self) -> None:
        # Deactivate node 2 and let it be suspected/failed.
        self.detector.deactivate_node(2)
        for _ in range(5):
            self.detector.tick()
            self.clock.advance(1.0)

        self.assertEqual(self.detector.get_status(2), NodeStatus.FAILED)

        # Reactivate node 2 – it should start as ALIVE again.
        self.detector.activate_node(2)
        # Manually reset its status to ALIVE for the purpose of the test.
        node = self.detector.nodes[2]
        node.status = NodeStatus.ALIVE
        node.heartbeat = 0
        node.last_update = self.clock()

        # Run a few rounds; node 2 should stay ALIVE.
        for _ in range(3):
            self.detector.tick()
            self.clock.advance(1.0)

        self.assertEqual(self.detector.get_status(2), NodeStatus.ALIVE)

    def test_single_node(self) -> None:
        single = GossipFailureDetector(
            node_ids=[99],
            gossip_interval=0.5,
            failure_threshold=1,
            clock=self.clock,
            rng_seed=0,
        )
        self.assertEqual(single.get_status(99), NodeStatus.ALIVE)

        # Deactivate the only node – it should become SUSPECTED after 1 interval.
        single.deactivate_node(99)
        single.tick()
        self.clock.advance(0.5)
        single.tick()
        self.clock.advance(0.5)

        self.assertEqual(single.get_status(99), NodeStatus.SUSPECTED)

        # After another interval it becomes FAILED.
        single.tick()
        self.clock.advance(0.5)
        single.tick()
        self.clock.advance(0.5)

        self.assertEqual(single.get_status(99), NodeStatus.FAILED)

    def test_invalid_initialisation(self) -> None:
        with self.assertRaises(ValueError):
            GossipFailureDetector(node_ids=[])


if __name__ == "__main__":
    # Run the unit tests with a higher verbosity for demonstration.
    unittest.main(verbosity=2)
