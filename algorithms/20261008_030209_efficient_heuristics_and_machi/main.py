import unittest
from typing import Set

from core import FaultCharacterizer


class TestFaultCharacterizer(unittest.TestCase):
    def setUp(self) -> None:
        self.fc = FaultCharacterizer()

    def test_tarantula_basic(self) -> None:
        # Two components A and B, three tests
        self.fc.add_test_result("t1", passed=False, coverage={"A"})
        self.fc.add_test_result("t2", passed=False, coverage={"A", "B"})
        self.fc.add_test_result("t3", passed=True, coverage={"B"})
        suspicious = self.fc.compute_suspiciousness()
        # Manual calculation:
        # total_failed = 2, total_passed = 1
        # A: failed=2, passed=0 -> s = (2/2) / ((2/2)+(0/1)) = 1
        # B: failed=1, passed=1 -> s = (1/2) / ((1/2)+(1/1)) = 0.5 / 1.5 = 1/3
        self.assertAlmostEqual(suspicious["A"], 1.0)
        self.assertAlmostEqual(suspicious["B"], 1.0 / 3.0)

    def test_naive_bayes_training_and_prediction(self) -> None:
        # Synthetic data: component X tends to be faulty when covered alone,
        # component Y when both X and Y are covered.
        self.fc.add_test_result(
            "t1", passed=False, coverage={"X"}, faulty_component="X"
        )
        self.fc.add_test_result(
            "t2", passed=False, coverage={"X", "Y"}, faulty_component="Y"
        )
        self.fc.add_test_result(
            "t3", passed=False, coverage={"Y"}, faulty_component="Y"
        )
        self.fc.train_naive_bayes()

        # Predict for a new failing test covering only X
        best, posterior = self.fc.predict_fault({"X"})
        self.assertEqual(best, "X")
        self.assertGreater(posterior["X"], posterior["Y"])

        # Predict for a failing test covering X and Y
        best2, posterior2 = self.fc.predict_fault({"X", "Y"})
        self.assertEqual(best2, "Y")
        self.assertGreater(posterior2["Y"], posterior2["X"])

    def test_predict_without_training(self) -> None:
        self.fc.add_test_result("t1", passed=False, coverage={"A"})
        with self.assertRaises(RuntimeError):
            self.fc.predict_fault({"A"})

    def test_clear_resets_state(self) -> None:
        self.fc.add_test_result("t1", passed=False, coverage={"A"}, faulty_component="A")
        self.fc.train_naive_bayes()
        self.fc.clear()
        self.assertEqual(self.fc.compute_suspiciousness(), {})
        with self.assertRaises(RuntimeError):
            self.fc.predict_fault({"A"})


def demo() -> None:
    """Simple interactive demonstration."""
    fc = FaultCharacterizer()
    # Add some test results
    fc.add_test_result("t1", passed=False, coverage={"mod1", "mod2"}, faulty_component="mod1")
    fc.add_test_result("t2", passed=False, coverage={"mod2"}, faulty_component="mod2")
    fc.add_test_result("t3", passed=True, coverage={"mod3"})
    # Heuristic
    print("Tarantula suspiciousness:")
    for comp, score in sorted(fc.compute_suspiciousness().items()):
        print(f"  {comp}: {score:.3f}")
    # Train classifier
    fc.train_naive_bayes()
    # Predict on a new failing test
    best, posterior = fc.predict_fault({"mod1", "mod3"})
    print(f"\nPredicted faulty component: {best}")
    print("Posterior distribution:")
    for comp, prob in sorted(posterior.items(), key=lambda kv: -kv[1]):
        print(f"  {comp}: {prob:.3f}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    print("\n--- Demo Run ---")
    demo()
