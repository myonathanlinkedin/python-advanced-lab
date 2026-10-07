import abc
import enum
import random
import uuid
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional


class ParameterType(enum.Enum):
    INTEGER = enum.auto()
    FLOAT = enum.auto()
    CATEGORICAL = enum.auto()


class Parameter:
    """Definition of a single hyperparameter."""

    def __init__(
        self,
        name: str,
        p_type: ParameterType,
        *,
        bounds: Tuple[Union[int, float], Union[int, float]] = (0, 1),
        choices: Sequence[Any] = (),
    ) -> None:
        self.name = name
        self.type = p_type
        if self.type == ParameterType.CATEGORICAL:
            if not choices:
                raise ValueError("Categorical parameters require non‑empty choices.")
            self.choices = list(choices)
        else:
            low, high = bounds
            if low >= high:
                raise ValueError("Lower bound must be < upper bound.")
            self.low = low
            self.high = high

    def sample(self, rng: random.Random) -> Any:
        if self.type == ParameterType.INTEGER:
            return rng.randint(int(self.low), int(self.high))
        if self.type == ParameterType.FLOAT:
            return rng.uniform(float(self.low), float(self.high))
        return rng.choice(self.choices)


class SearchSpace:
    """Container for a collection of Parameters."""

    def __init__(self, parameters: Sequence[Parameter]) -> None:
        if not parameters:
            raise ValueError("SearchSpace must contain at least one Parameter.")
        self.parameters = list(parameters)

    def sample(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        rng = rng or random.Random()
        return {p.name: p.sample(rng) for p in self.parameters}


class TrialStatus(enum.Enum):
    PENDING = enum.auto()
    COMPLETED = enum.auto()
    FAILED = enum.auto()


class Trial:
    """A single evaluation of a hyperparameter configuration."""

    def __init__(self, params: Dict[str, Any]) -> None:
        self.id: str = str(uuid.uuid4())
        self.params: Dict[str, Any] = params
        self.result: Optional[float] = None
        self.status: TrialStatus = TrialStatus.PENDING

    def complete(self, result: float) -> None:
        self.result = result
        self.status = TrialStatus.COMPLETED

    def fail(self) -> None:
        self.status = TrialStatus.FAILED


class Optimizer(abc.ABC):
    """Abstract base class for hyperparameter optimizers."""

    @abc.abstractmethod
    def suggest(self, n: int) -> List[Trial]:
        ...

    @abc.abstractmethod
    def observe(self, trial: Trial) -> None:
        ...


class RandomSearchOptimizer(Optimizer):
    """Simple random search optimizer."""

    def __init__(self, space: SearchSpace, seed: Optional[int] = None) -> None:
        self.space = space
        self.rng = random.Random(seed)
        self._suggested: set[str] = set()

    def suggest(self, n: int) -> List[Trial]:
        trials: List[Trial] = []
        while len(trials) < n:
            params = self.space.sample(self.rng)
            trial = Trial(params)
            # Ensure uniqueness of trial IDs (unlikely collision)
            if trial.id not in self._suggested:
                self._suggested.add(trial.id)
                trials.append(trial)
        return trials

    def observe(self, trial: Trial) -> None:
        # Random search does not adapt; method kept for interface compatibility.
        pass


class Vizier:
    """High‑level research interface for black‑box optimization."""

    def __init__(self, space: SearchSpace, optimizer: Optimizer) -> None:
        self.space = space
        self.optimizer = optimizer
        self._trials: Dict[str, Trial] = {}

    def suggest(self, n: int = 1) -> List[Trial]:
        new_trials = self.optimizer.suggest(n)
        for t in new_trials:
            self._trials[t.id] = t
        return new_trials

    def report(self, trial_id: str, result: float) -> None:
        trial = self._trials.get(trial_id)
        if trial is None:
            raise KeyError(f"Trial ID {trial_id} not found.")
        trial.complete(result)
        self.optimizer.observe(trial)

    def get_trial(self, trial_id: str) -> Trial:
        return self._trials[trial_id]

    def best_trial(self) -> Optional[Trial]:
        completed = [t for t in self._trials.values() if t.status == TrialStatus.COMPLETED]
        if not completed:
            return None
        return min(completed, key=lambda t: t.result)  # assume minimization
