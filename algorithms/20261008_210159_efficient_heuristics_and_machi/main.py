"""
main.py
---------

Unit‑test suite and demonstration entry‑point for the FaultClassifier.

The tests cover:
* Basic training / prediction workflow.
* Edge cases: single sample, k larger than dataset, constant features.
* Deterministic tie‑breaking.
* Accuracy scoring.

Running the module executes the test suite and a short demo.
"""

import unittest
from typing import List, Any

# Import the core implementation
from core import FaultClassifier


class TestFaultClassifier(unittest.TestCase):
    def setUp(self) -> None:
        # Simple synthetic dataset: two features, three fault categories
        self.X_train = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
        self.y_train = ["A", "A", "B", "B", "C"]

    def test_basic_fit_predict(self) -> None:
        clf = FaultClassifier(k=3)
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]])
        self.assertEqual(preds, ["A", "B", "C"])

    def test_k_larger_than_dataset(self) -> None:
        clf = FaultClassifier(k=10)  # larger than training size
        clf.fit(self.X_train, self.y_train)
        pred = clf.predict([[0.2, 0.2]])[0]
        # With all points considered, majority vote is A (2), B (2), C (1) -> tie A vs B
        # Deterministic tie‑break selects the smaller label alphabetically: "A"
        self.assertEqual(pred, "A")

    def test_constant_feature_scaling(self) -> None:
        # All samples share the same value for the second feature
        X_const = [[1, 5], [2, 5], [3, 5]]
        y_const = ["X", "Y", "Y"]
        clf = FaultClassifier(k=1, scale=True)
        clf.fit(X_const, y_const)
        pred = clf.predict([[2.5, 5]])[0]
        self.assertEqual(pred, "Y")  # nearest neighbour is the second sample

    def test_no_scaling(self) -> None:
        clf = FaultClassifier(k=1, scale=False)
        clf.fit(self.X_train, self.y_train)
        pred = clf.predict([[0.1, 0.1]])[0]
        self.assertEqual(pred, "A")

    def test_score_accuracy(self) -> None:
        clf = FaultClassifier(k=3)
        clf.fit(self.X_train, self.y_train)
        X_test = [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]
        y_test = ["A", "B", "C"]
        acc = clf.score(X_test, y_test)
        self.assertAlmostEqual(acc, 1.0)

    def test_empty_training_raises(self) -> None:
        clf = FaultClassifier()
        with self.assertRaises(ValueError):
            clf.fit([], [])

    def test_predict_before_fit_raises(self) -> None:
        clf = FaultClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict([[0, 0]])

    def test_invalid_k_raises(self) -> None:
        with self.assertRaises(ValueError):
            FaultClassifier(k=0)

    def test_mismatched_lengths_raise(self) -> None:
        clf = FaultClassifier()
        with self.assertRaises(ValueError):
            clf.fit([[0, 0], [1, 1]], ["only_one_label"])

    def test_tie_breaking_determinism(self) -> None:
        # Construct a scenario where two labels appear equally often among neighbours
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = ["B", "A", "A", "B"]
        clf = FaultClassifier(k=2)  # each point will have one A and one B neighbour
        clf.fit(X, y)
        pred = clf.predict([[0.5, 0.5]])[0]
        # Tie between "A" and "B" -> deterministic min() selects "A"
        self.assertEqual(pred, "A")


def demo() -> None:
    """A concise demonstration of the classifier on a tiny dataset."""
    X = [
        [2.0, 3.0],
        [1.0, 5.0],
        [2.5, 1.0],
        [3.0, 4.0],
        [0.5, 2.0],
    ]
    y = ["Leak", "Leak", "Short", "Short", "Open"]
    classifier = FaultClassifier(k=3)
    classifier.fit(X, y)

    test_samples = [
        [2.1, 3.1],
        [0.6, 2.1],
        [2.4, 1.2],
    ]
    predictions = classifier.predict(test_samples)
    for sample, pred in zip(test_samples, predictions):
        print(f"Sample {sample} -> Predicted fault: {pred}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    print("\n--- Demo ---")
    demo()
# End of main.py
