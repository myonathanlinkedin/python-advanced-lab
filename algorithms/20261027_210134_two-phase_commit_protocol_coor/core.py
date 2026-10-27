import enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional


class ParticipantState(enum.Enum):
    IDLE = enum.auto()
    PREPARED = enum.auto()
    COMMITTED = enum.auto()
    ABORTED = enum.auto()


class CoordinatorState(enum.Enum):
    IDLE = enum.auto()
    WAITING = enum.auto()
    COMMITTED = enum.auto()
    ABORTED = enum.auto()


class TwoPhaseError(RuntimeError):
    """Base class for 2PC protocol errors."""


class TimeoutError(TwoPhaseError):
    """Raised when a participant does not respond in time."""


@dataclass
class Participant:
    """A participant in a two‑phase commit protocol.

    Attributes
    ----------
    pid: int
        Unique participant identifier.
    vote_commit: bool
        Determines the vote this participant will cast during the prepare phase.
    fail_on_prepare: bool
        If True, the participant will raise a TimeoutError during prepare.
    state: ParticipantState
        Current state of the participant.
    """
    pid: int
    vote_commit: bool = True
    fail_on_prepare: bool = False
    state: ParticipantState = field(default=ParticipantState.IDLE, init=False)

    def prepare(self, txn_id: int) -> bool:
        """Handle a PREPARE request.

        Returns
        -------
        bool
            True if the participant votes to COMMIT, False to ABORT.
        """
        if self.state != ParticipantState.IDLE:
            raise TwoPhaseError(f"Participant {self.pid} not idle during prepare.")
        if self.fail_on_prepare:
            raise TimeoutError(f"Participant {self.pid} timed out on prepare.")
        self.state = ParticipantState.PREPARED
        return self.vote_commit

    def commit(self, txn_id: int) -> None:
        """Handle a COMMIT request."""
        if self.state != ParticipantState.PREPARED:
            raise TwoPhaseError(f"Participant {self.pid} cannot commit from state {self.state}.")
        self.state = ParticipantState.COMMITTED

    def abort(self, txn_id: int) -> None:
        """Handle an ABORT request."""
        if self.state not in (ParticipantState.PREPARED, ParticipantState.IDLE):
            raise TwoPhaseError(f"Participant {self.pid} cannot abort from state {self.state}.")
        self.state = ParticipantState.ABORTED

    def reset(self) -> None:
        """Return participant to IDLE state for a new transaction."""
        self.state = ParticipantState.IDLE

    def __repr__(self) -> str:
        return f"<Participant pid={self.pid} state={self.state.name}>"


@dataclass
class Coordinator:
    """Coordinator for the two‑phase commit protocol.

    Attributes
    ----------
    participants: List[Participant]
        Participants involved in the transaction.
    """
    participants: List[Participant]
    state: CoordinatorState = field(default=CoordinatorState.IDLE, init=False)
    txn_id_counter: int = field(default=0, init=False)

    def start_transaction(self, vote_overrides: Optional[Dict[int, bool]] = None,
                          fail_participants: Optional[List[int]] = None) -> None:
        """Execute a full two‑phase commit cycle.

        Parameters
        ----------
        vote_overrides: dict[int, bool] | None
            Optional mapping pid -> vote (True for commit, False for abort) to override
            each participant's default vote.
        fail_participants: list[int] | None
            Optional list of participant ids that will simulate a timeout during prepare.
        """
        if self.state != CoordinatorState.IDLE:
            raise TwoPhaseError("Coordinator already busy with a transaction.")
        self.txn_id_counter += 1
        txn_id = self.txn_id_counter
        self.state = CoordinatorState.WAITING

        # Apply overrides / failures for this run
        if vote_overrides:
            for p in self.participants:
                if p.pid in vote_overrides:
                    p.vote_commit = vote_overrides[p.pid]
        if fail_participants:
            for p in self.participants:
                p.fail_on_prepare = p.pid in fail_participants

        # Phase 1: Prepare
        votes = {}
        try:
            for p in self.participants:
                vote = p.prepare(txn_id)
                votes[p.pid] = vote
        except TimeoutError as te:
            # Any timeout forces abort
            self._abort(txn_id)
            raise TwoPhaseError(f"Transaction {txn_id} aborted due to timeout.") from te

        # Decision
        if all(votes.values()):
            self._commit(txn_id)
        else:
            self._abort(txn_id)

        # Reset temporary overrides
        for p in self.participants:
            p.fail_on_prepare = False

    def _commit(self, txn_id: int) -> None:
        """Send COMMIT to all participants and update coordinator state."""
        for p in self.participants:
            p.commit(txn_id)
        self.state = CoordinatorState.COMMITTED

    def _abort(self, txn_id: int) -> None:
        """Send ABORT to all participants and update coordinator state."""
        for p in self.participants:
            # Abort is safe even if participant never prepared
            try:
                p.abort(txn_id)
            except TwoPhaseError:
                # Ignore illegal aborts; they indicate participant was idle.
                pass
        self.state = CoordinatorState.ABORTED

    def reset(self) -> None:
        """Return coordinator and all participants to IDLE state."""
        self.state = CoordinatorState.IDLE
        for p in self.participants:
            p.reset()

    def __repr__(self) -> str:
        return f"<Coordinator state={self.state.name} txn_id={self.txn_id_counter}>"
