import unittest
from core import (
    Coordinator,
    Participant,
    Vote,
    ParticipantError,
    ParticipantState,
    CoordinatorState,
)


def always_commit(_: str) -> Vote:
    return Vote.COMMIT


def always_abort(_: str) -> Vote:
    return Vote.ABORT


def flaky_participant(_: str) -> Vote:
    raise ParticipantError("Simulated timeout")


class TestTwoPhaseCommit(unittest.TestCase):
    def test_successful_commit(self):
        coord = Coordinator()
        p1 = Participant("p1", always_commit)
        p2 = Participant("p2", always_commit)
        coord.add_participant(p1)
        coord.add_participant(p2)

        final_state, log = coord.run()
        self.assertEqual(final_state, CoordinatorState.COMMITTED)
        self.assertEqual(p1.state, ParticipantState.COMMITTED)
        self.assertEqual(p2.state, ParticipantState.COMMITTED)
        self.assertIn("All participants voted COMMIT – sending COMMIT", log)

    def test_abort_due_to_vote(self):
        coord = Coordinator()
        p1 = Participant("p1", always_commit)
        p2 = Participant("p2", always_abort)
        coord.add_participant(p1)
        coord.add_participant(p2)

        final_state, log = coord.run()
        self.assertEqual(final_state, CoordinatorState.ABORTED)
        self.assertEqual(p1.state, ParticipantState.ABORTED)
        self.assertEqual(p2.state, ParticipantState.ABORTED)
        self.assertIn("Aborting transaction – sending ABORT", log)

    def test_abort_due_to_timeout(self):
        coord = Coordinator()
        p1 = Participant("p1", always_commit)
        p2 = Participant("p2", flaky_participant)
        coord.add_participant(p1)
        coord.add_participant(p2)

        final_state, log = coord.run()
        self.assertEqual(final_state, CoordinatorState.ABORTED)
        self.assertEqual(p1.state, ParticipantState.ABORTED)
        self.assertEqual(p2.state, ParticipantState.INIT)  # never left INIT due to timeout
        self.assertTrue(any("Simulated timeout" in entry for entry in log))

    def test_invalid_state_transition(self):
        p = Participant("p", always_commit)
        # Directly invoke commit without prepare – should raise RuntimeError
        with self.assertRaises(RuntimeError):
            p.receive_commit()
        # After proper prepare, commit works
        p.receive_prepare()
        p.receive_commit()
        self.assertEqual(p.state, ParticipantState.COMMITTED)

    def test_coordinator_rejects_late_participant_addition(self):
        coord = Coordinator()
        p = Participant("p", always_commit)
        coord.add_participant(p)
        coord.run()
        with self.assertRaises(RuntimeError):
            coord.add_participant(Participant("q", always_commit))


if __name__ == "__main__":
    # Simple demonstration
    demo_coord = Coordinator()
    demo_coord.add_participant(Participant("A", always_commit))
    demo_coord.add_participant(Participant("B", always_commit))
    state, demo_log = demo_coord.run()
    print(f"Demo finished with coordinator state: {state.value}")
    for line in demo_log:
        print(line)

    # Run unit tests
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
