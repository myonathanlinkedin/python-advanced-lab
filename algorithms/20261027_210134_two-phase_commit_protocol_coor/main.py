import unittest
from core import Participant, Coordinator, TwoPhaseError, TimeoutError, CoordinatorState, ParticipantState


class TestTwoPhaseCommit(unittest.TestCase):
    def setUp(self) -> None:
        self.participants = [Participant(pid=i) for i in range(3)]
        self.coord = Coordinator(participants=self.participants)

    def tearDown(self) -> None:
        self.coord.reset()

    def test_successful_commit(self):
        """All participants vote commit; transaction should commit."""
        self.coord.start_transaction()
        self.assertEqual(self.coord.state, CoordinatorState.COMMITTED)
        for p in self.participants:
            self.assertEqual(p.state, ParticipantState.COMMITTED)

    def test_abort_due_to_vote(self):
        """One participant votes abort; transaction should abort."""
        overrides = {1: False}  # participant 1 votes abort
        self.coord.start_transaction(vote_overrides=overrides)
        self.assertEqual(self.coord.state, CoordinatorState.ABORTED)
        for p in self.participants:
            self.assertEqual(p.state, ParticipantState.ABORTED)

    def test_abort_due_to_timeout(self):
        """A participant times out during prepare; transaction aborts."""
        fail_ids = [2]  # participant 2 will timeout
        with self.assertRaises(TwoPhaseError) as cm:
            self.coord.start_transaction(fail_participants=fail_ids)
        self.assertIn("aborted due to timeout", str(cm.exception))
        self.assertEqual(self.coord.state, CoordinatorState.ABORTED)
        # Participants that prepared before timeout should be aborted as well
        for p in self.participants:
            self.assertEqual(p.state, ParticipantState.ABORTED)

    def test_multiple_transactions(self):
        """Coordinator can run successive transactions after reset."""
        # First transaction commits
        self.coord.start_transaction()
        self.assertEqual(self.coord.state, CoordinatorState.COMMITTED)
        # Reset and run a second transaction where a vote aborts
        self.coord.reset()
        overrides = {0: False}
        self.coord.start_transaction(vote_overrides=overrides)
        self.assertEqual(self.coord.state, CoordinatorState.ABORTED)
        # Verify final states
        for p in self.participants:
            self.assertEqual(p.state, ParticipantState.ABORTED)

    def test_participant_illegal_transition(self):
        """Participant raises error if commit called without prepare."""
        p = self.participants[0]
        with self.assertRaises(TwoPhaseError):
            p.commit(txn_id=1)


if __name__ == "__main__":
    # Simple demonstration
    participants = [Participant(pid=i, vote_commit=True) for i in range(2)]
    coordinator = Coordinator(participants=participants)

    print("=== Demo: Successful Commit ===")
    coordinator.start_transaction()
    print(coordinator)
    for p in participants:
        print(p)

    coordinator.reset()
    print("\n=== Demo: Abort due to Vote ===")
    coordinator.start_transaction(vote_overrides={1: False})
    print(coordinator)
    for p in participants:
        print(p)

    # Run unit tests
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
