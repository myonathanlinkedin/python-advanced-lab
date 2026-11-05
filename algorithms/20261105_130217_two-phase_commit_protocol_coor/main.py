import unittest
from typing import List

from core import Participant, Coordinator, ParticipantState, Vote


class TestTwoPhaseCommit(unittest.TestCase):
    def setUp(self) -> None:
        # Helper to create participants with custom behaviour.
        self.make_participants = lambda specs: [
            Participant(name=spec["name"], will_commit=spec.get("will_commit", True),
                        fail_on_prepare=spec.get("fail_on_prepare", False))
            for spec in specs
        ]

    def test_successful_commit(self) -> None:
        specs = [
            {"name": "P1"},
            {"name": "P2"},
            {"name": "P3"},
        ]
        participants: List[Participant] = self.make_participants(specs)
        coord = Coordinator(participants)

        outcome = coord.run_transaction()
        self.assertTrue(outcome, "Transaction should commit when all votes are COMMIT")

        for p in participants:
            self.assertEqual(p.state, ParticipantState.COMMITTED,
                             f"{p.name} should be COMMITTED")

    def test_abort_due_to_negative_vote(self) -> None:
        specs = [
            {"name": "P1"},
            {"name": "P2", "will_commit": False},  # votes ABORT
            {"name": "P3"},
        ]
        participants = self.make_participants(specs)
        coord = Coordinator(participants)

        outcome = coord.run_transaction()
        self.assertFalse(outcome, "Transaction should abort when any vote is ABORT")

        for p in participants:
            self.assertEqual(p.state, ParticipantState.ABORTED,
                             f"{p.name} should be ABORTED after abort path")

    def test_abort_due_to_prepare_exception(self) -> None:
        specs = [
            {"name": "P1"},
            {"name": "P2", "fail_on_prepare": True},  # raises during prepare
            {"name": "P3"},
        ]
        participants = self.make_participants(specs)
        coord = Coordinator(participants)

        outcome = coord.run_transaction()
        self.assertFalse(outcome, "Transaction should abort when a participant fails during prepare")

        for p in participants:
            self.assertEqual(p.state, ParticipantState.ABORTED,
                             f"{p.name} should be ABORTED after exception path")

    def test_coordinator_reuse_not_allowed(self) -> None:
        specs = [
            {"name": "P1"},
            {"name": "P2"},
        ]
        participants = self.make_participants(specs)
        coord = Coordinator(participants)

        # First transaction succeeds.
        self.assertTrue(coord.run_transaction())

        # Second transaction should raise because participants are no longer INIT.
        with self.assertRaises(RuntimeError):
            coord.run_transaction()

    def test_participant_invalid_state_transitions(self) -> None:
        p = Participant("Solo")
        # Commit without prepare should fail.
        with self.assertRaises(RuntimeError):
            p.commit()

        # Abort from INIT is allowed.
        p.abort()
        self.assertEqual(p.state, ParticipantState.ABORTED)

        # Subsequent abort is idempotent.
        p.abort()
        self.assertEqual(p.state, ParticipantState.ABORTED)

        # Prepare after abort is illegal.
        with self.assertRaises(RuntimeError):
            p.prepare()


if __name__ == "__main__":
    unittest.main()
