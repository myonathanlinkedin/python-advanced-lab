import unittest
import math
from typing import List

from core import (
    Parameter,
    ParameterType,
    SearchSpace,
    RandomSearchOptimizer,
    Vizier,
    Trial,
    TrialStatus,
)


def quadratic_objective(params: dict) -> float:
    """Simple convex function: (x-3)^2 + (y+1)^2."""
    x = params["x"]
    y = params["y"]
    return (x - 3) ** 2 + (y + 1) ** 2


class TestVizier(unittest.TestCase):
    def setUp(self) -> None:
        self.space = SearchSpace(
            [
                Parameter("x", ParameterType.FLOAT, bounds=(-10.0, 10.0)),
                Parameter("y", ParameterType.FLOAT, bounds=(-10.0, 10.0)),
                Parameter("z", ParameterType.CATEGORICAL, choices=["a", "b", "c"]),
            ]
        )
        self.optimizer = RandomSearchOptimizer(self.space, seed=42)
        self.vizier = Vizier(self.space, self.optimizer)

    def test_parameter_sampling(self) -> None:
        sample = self.space.sample()
        self.assertIn("x", sample)
        self.assertIn("y", sample)
        self.assertIn("z", sample)
        self.assertGreaterEqual(sample["x"], -10.0)
        self.assertLessEqual(sample["x"], 10.0)
        self.assertIn(sample["z"], ["a", "b", "c"])

    def test_suggest_and_report(self) -> None:
        trials: List[Trial] = self.vizier.suggest(3)
        self.assertEqual(len(trials), 3)
        for t in trials:
            self.assertEqual(t.status, TrialStatus.PENDING)
            # evaluate and report
            result = quadratic_objective(t.params)
            self.vizier.report(t.id, result)
            self.assertEqual(t.status, TrialStatus.COMPLETED)
            self.assertIsNotNone(t.result)

    def test_best_trial(self) -> None:
        # Run a small optimization loop
        for _ in range(30):
            trials = self.vizier.suggest(5)
            for t in trials:
                self.vizier.report(t.id, quadratic_objective(t.params))
        best = self.vizier.best_trial()
        self.assertIsNotNone(best)
        self.assertIsInstance(best, Trial)
        # The optimum of the quadratic is at (3, -1); allow tolerance
        self.assertAlmostEqual(best.params["x"], 3, delta=2.0)
        self.assertAlmostEqual(best.params["y"], -1, delta=2.0)

    def test_invalid_report(self) -> None:
        with self.assertRaises(KeyError):
            self.vizier.report("nonexistent-id", 0.0)


if __name__ == "__main__":
    # Simple demo run outside unittest
    space = SearchSpace(
        [
            Parameter("x", ParameterType.FLOAT, bounds=(-5, 5)),
            Parameter("y", ParameterType.FLOAT, bounds=(-5, 5)),
        ]
    )
    optimizer = RandomSearchOptimizer(space, seed=123)
    vizier = Vizier(space, optimizer)

    for _ in range(10):
        trial = vizier.suggest(1)[0]
        result = quadratic_objective(trial.params)
        vizier.report(trial.id, result)
        print(f"Trial {trial.id[:8]} params={trial.params} result={result:.4f}")

    best = vizier.best_trial()
    if best:
        print(
            f"\nBest trial: {best.id[:8]} params={best.params} result={best.result:.4f}"
        )

    # Run unit tests
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
