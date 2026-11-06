import enum
from typing import Callable, Dict, List, Tuple


class Vote(enum.Enum):
    COMMIT = "COMMIT"
    ABORT = "ABORT"


class ParticipantState(enum.Enum):
    INIT = "INIT"
    READY = "READY"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


class CoordinatorState(enum.Enum):
    INIT = "INIT"
    WAITING = "WAITING"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


class ParticipantError(Exception):
    """Raised when a participant cannot process a request (e.g., timeout)."""


class Participant:
    """
    A participant in a Two‑Phase Commit protocol.

    The participant decides its vote via a user‑supplied callable `vote_func`.
    The callable receives the participant id and must return a :class:`Vote`.
    It may raise :class:`ParticipantError` to simulate a timeout or failure.
    """

    def __init__(self, participant_id: str, vote_func: Callable[[str], Vote]) -> None:
        self.id: str = participant_id
        self._vote_func: Callable[[str], Vote] = vote_func
        self.state: ParticipantState = ParticipantState.INIT

    def receive_prepare(self) -> Vote:
        """Handle the PREPARE request from the coordinator."""
        if self.state is not ParticipantState.INIT:
            raise RuntimeError(f"Participant {self.id} received PREPARE in invalid state {self.state}")

        try:
            vote = self._vote_func(self.id)
        except Exception as exc:
            raise ParticipantError(f"Participant {self.id} failed during prepare: {exc}") from exc

        if vote is Vote.COMMIT:
            self.state = ParticipantState.READY
        else:
            self.state = ParticipantState.ABORTED
        return vote

    def receive_commit(self) -> None:
        """Handle the COMMIT request from the coordinator."""
        if self.state is not ParticipantState.READY:
            raise RuntimeError(f"Participant {self.id} cannot COMMIT from state {self.state}")
        self.state = ParticipantState.COMMITTED

    def receive_abort(self) -> None:
        """Handle the ABORT request from the coordinator."""
        if self.state not in (ParticipantState.READY, ParticipantState.INIT):
            raise RuntimeError(f"Participant {self.id} cannot ABORT from state {self.state}")
        self.state = ParticipantState.ABORTED

    def __repr__(self) -> str:
        return f"<Participant id={self.id} state={self.state.value}>"


class Coordinator:
    """
    Coordinator for the Two‑Phase Commit protocol.

    It manages a collection of participants, drives the protocol,
    and records the final outcome.
    """

    def __init__(self) -> None:
        self._participants: List[Participant] = []
        self.state: CoordinatorState = CoordinatorState.INIT
        self._log: List[str] = []

    def add_participant(self, participant: Participant) -> None:
        if self.state is not CoordinatorState.INIT:
            raise RuntimeError("Cannot add participants after protocol start")
        self._participants.append(participant)
        self._log.append(f"Added participant {participant.id}")

    def run(self) -> Tuple[CoordinatorState, List[str]]:
        """
        Execute the full Two‑Phase Commit protocol.

        Returns:
            A tuple of final coordinator state and the event log.
        """
        if not self._participants:
            raise RuntimeError("No participants registered")
        self.state = CoordinatorState.WAITING
        self._log.append("Coordinator entering WAITING state")
        votes: Dict[str, Vote] = {}

        # Phase 1 – Prepare
        for p in self._participants:
            try:
                vote = p.receive_prepare()
                votes[p.id] = vote
                self._log.append(f"Participant {p.id} voted {vote.value}")
            except ParticipantError as exc:
                self._log.append(str(exc))
                self._abort()
                return self.state, self._log

        # Decision
        if all(v is Vote.COMMIT for v in votes.values()):
            self._commit()
        else:
            self._abort()
        return self.state, self._log

    def _commit(self) -> None:
        self._log.append("All participants voted COMMIT – sending COMMIT")
        for p in self._participants:
            try:
                p.receive_commit()
                self._log.append(f"Participant {p.id} COMMITTED")
            except Exception as exc:
                self._log.append(f"Commit failure on {p.id}: {exc}")
        self.state = CoordinatorState.COMMITTED
        self._log.append("Coordinator state -> COMMITTED")

    def _abort(self) -> None:
        self._log.append("Aborting transaction – sending ABORT")
        for p in self._participants:
            try:
                p.receive_abort()
                self._log.append(f"Participant {p.id} ABORTED")
            except Exception as exc:
                self._log.append(f"Abort failure on {p.id}: {exc}")
        self.state = CoordinatorState.ABORTED
        self._log.append("Coordinator state -> ABORTED")

    def __repr__(self) -> str:
        return f"<Coordinator state={self.state.value} participants={len(self._participants)}>"
