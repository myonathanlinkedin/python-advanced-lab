from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
import math


@dataclass(frozen=True)
class TestResult:
    """Immutable record of a single test execution."""
    test_id: str
    passed: bool
    coverage: Set[str]  # components exercised by the test
    faulty_component: Optional[str] = None  # ground‑truth for training


class FaultCharacterizer:
    """
    Provides heuristic (Tarantula) and a simple Naïve Bayes classifier
    for fault localization based on test coverage information.
    """

    def __init__(self) -> None:
        self._results: List[TestResult] = []
        self._components: Set[str] = set()
        # Naïve Bayes model parameters
        self._priors: Dict[str, float] = {}
        self._likelihoods: Dict[str, Dict[str, float]] = {}
        self._trained: bool = False

    # --------------------------------------------------------------------- #
    # Data collection
    # --------------------------------------------------------------------- #
    def add_test_result(
        self,
        test_id: str,
        passed: bool,
        coverage: Set[str],
        faulty_component: Optional[str] = None,
    ) -> None:
        """Store a test execution record."""
        result = TestResult(test_id, passed, frozenset(coverage), faulty_component)
        self._results.append(result)
        self._components.update(coverage)

    # --------------------------------------------------------------------- #
    # Heuristic: Tarantula suspiciousness
    # --------------------------------------------------------------------- #
    def compute_suspiciousness(self) -> Dict[str, float]:
        """
        Compute Tarantula suspiciousness for each component.

        s(c) = (failed_c / total_failed) /
               ((failed_c / total_failed) + (passed_c / total_passed))

        Returns a mapping component -> suspiciousness (0.0 … 1.0).
        """
        total_failed = sum(not r.passed for r in self._results)
        total_passed = sum(r.passed for r in self._results)

        if total_failed == 0 or total_passed == 0:
            # Degenerate case: return zero for all components
            return {c: 0.0 for c in self._components}

        failed_counts: Dict[str, int] = {c: 0 for c in self._components}
        passed_counts: Dict[str, int] = {c: 0 for c in self._components}

        for r in self._results:
            target = failed_counts if not r.passed else passed_counts
            for comp in r.coverage:
                target[comp] += 1

        suspiciousness: Dict[str, float] = {}
        for comp in self._components:
            f = failed_counts[comp] / total_failed
            p = passed_counts[comp] / total_passed
            denom = f + p
            suspiciousness[comp] = f / denom if denom > 0 else 0.0
        return suspiciousness

    # --------------------------------------------------------------------- #
    # Naïve Bayes classifier (binary features)
    # --------------------------------------------------------------------- #
    def train_naive_bayes(self) -> None:
        """
        Train a simple multinomial Naïve Bayes classifier.
        Each component is a possible class (faulty component).
        Features are binary presence/absence of components in coverage.
        """
        # Filter training data: we need a known faulty component
        training = [r for r in self._results if r.faulty_component]
        if not training:
            raise ValueError("No training data with known faulty components.")

        class_counts: Dict[str, int] = {}
        feature_counts: Dict[str, Dict[str, int]] = {}

        for r in training:
            cls = r.faulty_component
            class_counts[cls] = class_counts.get(cls, 0) + 1
            if cls not in feature_counts:
                feature_counts[cls] = {c: 0 for c in self._components}
            for comp in self._components:
                if comp in r.coverage:
                    feature_counts[cls][comp] += 1

        total_instances = sum(class_counts.values())
        self._priors = {
            cls: count / total_instances for cls, count in class_counts.items()
        }

        # Laplace smoothing factor
        alpha = 1.0
        self._likelihoods = {}
        for cls, feats in feature_counts.items():
            total_feat = sum(feats.values())
            denom = total_feat + alpha * len(self._components)
            self._likelihoods[cls] = {
                comp: (feats[comp] + alpha) / denom for comp in self._components
            }

        self._trained = True

    def predict_fault(self, coverage: Set[str]) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Predict the most likely faulty component given a coverage set.
        Returns a tuple (best_component, posterior_distribution).
        If the model is not trained, raises RuntimeError.
        """
        if not self._trained:
            raise RuntimeError("Naïve Bayes model has not been trained.")

        log_post: Dict[str, float] = {}
        for cls in self._priors:
            log_prob = math.log(self._priors[cls])
            for comp in self._components:
                prob = self._likelihoods[cls][comp]
                if comp in coverage:
                    log_prob += math.log(prob)
                else:
                    log_prob += math.log(1.0 - prob)
            log_post[cls] = log_prob

        # Convert log‑probabilities to normalized probabilities
        max_log = max(log_post.values())
        exp_shifted = {cls: math.exp(lp - max_log) for cls, lp in log_post.items()}
        total = sum(exp_shifted.values())
        posterior = {cls: val / total for cls, val in exp_shifted.items()}

        best = max(posterior, key=posterior.get) if posterior else None
        return best, posterior

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #
    def clear(self) -> None:
        """Reset all stored data and trained model."""
        self._results.clear()
        self._components.clear()
        self._priors.clear()
        self._likelihoods.clear()
        self._trained = False
