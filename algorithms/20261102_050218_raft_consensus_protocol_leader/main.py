from __future__ import annotations
import unittest

from engine import RaftCluster, NodeState


class TestRaftLeaderElection(unittest.TestCase):
    def setUp(self) -> None:
        # Create a deterministic 5‑node cluster
        self.cluster = RaftCluster(node_ids=[1, 2, 3, 4, 5])

    def test_initial_state_no_leader(self) -> None:
        self.assertIsNone(self.cluster.get_leader(), "There should be no leader initially")

    def test_eventual_leader_election(self) -> None:
        # Run enough ticks for at least one election timeout to fire
        for _ in range(30):
            self.cluster.tick_all()
        leader = self.cluster.get_leader()
        self.assertIsNotNone(leader, "A leader should have been elected")
        self.assertEqual(leader.state, NodeState.LEADER)
        # All other nodes must be followers and share the same term
        term = leader.current_term
        for node in self.cluster.nodes.values():
            if node.id == leader.id:
                continue
            self.assertTrue(node.active, "All nodes should be active")
            self.assertEqual(node.state, NodeState.FOLLOWER)
            self.assertEqual(node.current_term, term)

    def test_leader_failure_triggers_new_election(self) -> None:
        # First, let a leader emerge
        for _ in range(30):
            self.cluster.tick_all()
        old_leader = self.cluster.get_leader()
        self.assertIsNotNone(old_leader, "Initial leader expected")
        old_term = old_leader.current_term

        # Simulate crash of the leader
        self.cluster.deactivate_node(old_leader.id)

        # Run enough ticks for remaining nodes to notice missing heartbeats
        for _ in range(30):
            self.cluster.tick_all()

        new_leader = self.cluster.get_leader()
        self.assertIsNotNone(new_leader, "A new leader should be elected after crash")
        self.assertNotEqual(new_leader.id, old_leader.id, "New leader must be different")
        self.assertGreater(new_leader.current_term, old_term, "Term must increase after new election")

        # Reactivate old leader; it should step down to follower
        self.cluster.activate_node(old_leader.id)
        for _ in range(5):
            self.cluster.tick_all()
        self.assertEqual(old_leader.state, NodeState.FOLLOWER)
        self.assertEqual(old_leader.current_term, new_leader.current_term)

    def test_split_vote_no_multiple_leaders(self) -> None:
        """
        Force a scenario where two candidates could receive votes but not a majority.
        The deterministic timeouts ensure that only one candidate times out first,
        preventing split votes. The test validates that at most one leader exists.
        """
        # Run many ticks; the deterministic timeouts guarantee a single leader.
        for _ in range(100):
            self.cluster.tick_all()
        leader = self.cluster.get_leader()
        self.assertIsNotNone(leader, "Leader must exist")
        # Verify uniqueness
        leaders = [n for n in self.cluster.nodes.values() if n.state == NodeState.LEADER]
        self.assertEqual(len(leaders), 1, "Exactly one leader should be present")

    def test_term_monotonicity(self) -> None:
        """
        Ensure that term numbers never decrease across the whole cluster.
        """
        previous_terms = {nid: 0 for nid in self.cluster.nodes}
        for _ in range(50):
            self.cluster.tick_all()
            for nid, node in self.cluster.nodes.items():
                self.assertGreaterEqual(node.current_term, previous_terms[nid],
                                        f"Node {nid} term decreased")
                previous_terms[nid] = node.current_term


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
