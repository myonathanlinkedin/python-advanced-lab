from __future__ import annotations

from enum import Enum, auto
from typing import List, Tuple


class ParticipantState(Enum):
    """Possible states of a participant in the 2PC protocol."""
    INIT = auto()
    PREPARED = auto()
    COMMITTED = auto()
    ABORTED = auto()


class Vote(Enum):
    """Vote returned by a participant during the prepare phase."""
    COMMIT = auto()
    ABORT = auto()


class Participant:
    """
    A simulated participant in a two‑phase commit protocol.

    The participant can be configured to:
      * vote to commit (default) or abort,
      * raise an exception during the prepare phase to simulate a crash.
    """

    def __init__(
        self,
        name: str,
        *,
        will_commit: bool = True,
        fail_on_prepare: bool = False,
    ) -> None:
        self.name: str = name
        self.will_commit: bool = will_commit
        self.fail_on_prepare: bool = fail_on_prepare
        self.state: ParticipantState = ParticipantState.INIT

    # --------------------------------------------------------------------- #
    # Phase‑1: Prepare
    # --------------------------------------------------------------------- #
    def prepare(self) -> Vote:
        """
        Execute the prepare step.

        Returns:
            Vote.COMMIT if the participant is willing to commit,
            Vote.ABORT otherwise.

        Raises:
            RuntimeError: if ``fail_on_prepare`` is True, simulating a crash.
        """
        if self.state is not ParticipantState.INIT:
            raise RuntimeError(f"{self.name}: prepare called in invalid state {self.state}")

        if self.fail_on_prepare:
            raise RuntimeError(f"{self.name}: simulated failure during prepare")

        # Transition to PREPARED regardless of the vote – this matches the
        # classic 2PC semantics where a participant records its intent.
        self.state = ParticipantState.PREPARED
        return Vote.COMMIT if self.will_commit else Vote.ABORT

    # --------------------------------------------------------------------- #
    # Phase‑2: Commit / Abort
    # --------------------------------------------------------------------- #
    def commit(self) -> None:
        """
        Finalize the transaction with a commit.

        Raises:
            RuntimeError: if called when the participant is not in PREPARED state.
        """
        if self.state is not ParticipantState.PREPARED:
            raise RuntimeError(f"{self.name}: commit called in invalid state {self.state}")
        self.state = ParticipantState.COMMITTED

    def abort(self) -> None:
        """
        Finalize the transaction with an abort.

        This method is idempotent: calling abort multiple times leaves the
        participant in the ABORTED state.
        """
        # Abort can be called from INIT (e.g., if prepare never succeeded) or
        # from PREPARED. Any other state is a programming error.
        if self.state not in (ParticipantState.INIT, ParticipantState.PREPARED):
            raise RuntimeError(f"{self.name}: abort called in invalid state {self.state}")
        self.state = ParticipantState.ABORTED

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        return f"<Participant {self.name!r} state={self.state.name}>"


class Coordinator:
    """
    Coordinator for the two‑phase commit protocol.

    It orchestrates a single transaction across a fixed set of participants.
    The coordinator itself is stateless between transactions; however, it
    validates that all participants are in the INIT state before starting.
    """

    def __init__(self, participants: List[Participant]) -> None:
        if not participants:
            raise ValueError("Coordinator requires at least one participant")
        self.participants: List[Participant] = participants

    def _check_participants_ready(self) -> None:
        """Ensure every participant is in the INIT state before a new transaction."""
        for p in self.participants:
            if p.state is not ParticipantState.INIT:
                raise RuntimeError(
                    f"Participant {p.name} not ready (state={p.state}); "
                    "cannot start a new transaction."
                )

    def run_transaction(self) -> bool:
        """
        Execute a full two‑phase commit transaction.

        Returns:
            True if the transaction was committed, False if aborted.

        The method follows these steps:
          1. Verify all participants are ready (INIT).
          2. Phase‑1: ask each participant to prepare and collect votes.
          3. Phase‑2: if all votes are COMMIT, send commit; otherwise send abort.
        """
        self._check_participants_ready()

        # ---------- Phase 1 ----------
        votes: List[Tuple[Participant, Vote]] = []
        abort_needed = False

        for participant in self.participants:
            try:
                vote = participant.prepare()
                votes.append((participant, vote))
                if vote is Vote.ABORT:
                    abort_needed = True
            except Exception:
                # Any exception during prepare forces an abort.
                abort_needed = True
                # Record a dummy abort vote for completeness.
                votes.append((participant, Vote.ABORT))

        # ---------- Phase 2 ----------
        if abort_needed:
            for participant, _ in votes:
                # Participants that never reached PREPARED may still be in INIT.
                # abort() handles both cases.
                try:
                    participant.abort()
                except Exception as exc:
                    # In a production system we would log this; here we re‑raise.
                    raise RuntimeError(
                        f"Failed to abort participant {participant.name}"
                    ) from exc
            return False
        else:
            for participant, _ in votes:
                participant.commit()
            return True
