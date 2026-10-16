"""
core.py
---------

Implementation of a lightweight, pure‑Python k‑Nearest Neighbors (k‑NN) classifier
tailored for fault characterization tasks.  The module provides:

* `FaultClassifier` – a deterministic, deterministic, standard‑library only
  classifier with `fit`, `predict`, and `score` methods.
* Helper utilities for feature scaling and distance computation.
* Full type annotations and exhaustive docstrings.

The implementation avoids any external dependencies and is suitable for unit‑testing
and educational purposes while adhering to strict algorithmic rigor.
"""

from __future__ import annotations

import math
import itertools
from typing import List, Tuple, Sequence, Any, Callable, Iterable, Dict, Optional

Number = float | int


def _euclidean_distance(a: Sequence[Number], b: Sequence[Number]) -> float:
    """Return the Euclidean distance between two equal‑length numeric vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must be of the same length")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _min_max_scale(
    data: List[Sequence[Number]],
) -> Tuple[List[List[float]], List[Tuple[float, float]]]:
    """
    Scale each feature to the [0, 1] interval using min‑max scaling.

    Returns a tuple ``(scaled_data, feature_ranges)`` where
    ``feature_ranges[i]`` is ``(min_i, max_i)`` for the i‑th feature.
    """
    if not data:
        raise ValueError("Cannot scale an empty dataset")
    # Transpose to work column‑wise
    transposed = list(itertools.zip_longest(*data, fillvalue=0.0))
    ranges: List[Tuple[float, float]] = []
    scaled_columns: List[List[float]] = []

    for col in transposed:
        col_floats = [float(v) for v in col]
        col_min = min(col_floats)
        col_max = max(col_floats)
        ranges.append((col_min, col_max))
        if math.isclose(col_max, col_min):
            # Constant column – map everything to 0.0 to avoid division by zero
            scaled_columns.append([0.0 for _ in col_floats])
        else:
            scaled_columns.append(
                [(v - col_min) / (col_max - col_min) for v in col_floats]
            )

    # Transpose back to row‑wise representation
    scaled_data = [list(row) for row in zip(*scaled_columns)]
    return scaled_data, ranges


def _apply_min_max_scale(
    sample: Sequence[Number], ranges: List[Tuple[float, float]]
) -> List[float]:
    """Scale a single sample using previously computed ``ranges``."""
    if len(sample) != len(ranges):
        raise ValueError("Sample length does not match number of feature ranges")
    scaled = []
    for value, (col_min, col_max) in zip(sample, ranges):
        if math.isclose(col_max, col_min):
            scaled.append(0.0)
        else:
            scaled.append((float(value) - col_min) / (col_max - col_min))
    return scaled


class FaultClassifier:
    """
    A deterministic k‑Nearest Neighbors classifier for fault characterization.

    Parameters
    ----------
    k : int, optional
        Number of neighbours to consider. Must be >= 1. Default is 3.
    distance : Callable[[Sequence[Number], Sequence[Number]], float], optional
        Distance metric; defaults to Euclidean distance.
    scale : bool, optional
        Whether to apply min‑max scaling to features. Default is True.
    """

    def __init__(
        self,
        k: int = 3,
        distance: Callable[[Sequence[Number], Sequence[Number]], float] = _euclidean_distance,
        scale: bool = True,
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.distance = distance
        self.scale = scale
        self._trained: bool = False
        self._X: List[List[float]] = []
        self._y: List[Any] = []
        self._ranges: Optional[List[Tuple[float, float]]] = None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def fit(self, X: Sequence[Sequence[Number]], y: Sequence[Any]) -> "FaultClassifier":
        """
        Store the training data.

        Parameters
        ----------
        X : sequence of feature vectors
        y : sequence of labels (same length as ``X``)

        Returns
        -------
        self
        """
        X_list = [list(map(float, row)) for row in X]
        y_list = list(y)

        if len(X_list) != len(y_list):
            raise ValueError("Feature matrix and label vector must have the same length")
        if not X_list:
            raise ValueError("Training data cannot be empty")

        if self.scale:
            X_scaled, ranges = _min_max_scale(X_list)
            self._X = X_scaled
            self._ranges = ranges
        else:
            self._X = X_list
            self._ranges = None

        self._y = y_list
        self._trained = True
        return self

    def predict(self, X: Sequence[Sequence[Number]]) -> List[Any]:
        """
        Predict the label for each sample in ``X``.

        Parameters
        ----------
        X : sequence of feature vectors

        Returns
        -------
        List of predicted labels.
        """
        if not self._trained:
            raise RuntimeError("Classifier has not been fitted yet")
        X_list = [list(map(float, row)) for row in X]

        if self.scale:
            if self._ranges is None:
                raise RuntimeError("Scaling ranges missing despite scale=True")
            X_proc = [_apply_min_max_scale(row, self._ranges) for row in X_list]
        else:
            X_proc = X_list

        predictions: List[Any] = []
        for sample in X_proc:
            predictions.append(self._predict_one(sample))
        return predictions

    def score(self, X: Sequence[Sequence[Number]], y_true: Sequence[Any]) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : sequence of feature vectors
        y_true : true labels

        Returns
        -------
        Accuracy as a float in [0, 1].
        """
        y_pred = self.predict(X)
        if len(y_true) != len(y_pred):
            raise ValueError("Length mismatch between true and predicted labels")
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        return correct / len(y_true) if y_true else 0.0

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _predict_one(self, sample: Sequence[float]) -> Any:
        """Predict label for a single pre‑processed sample."""
        # Compute distances to all training points
        distances = [
            (self.distance(sample, train_vec), label)
            for train_vec, label in zip(self._X, self._y)
        ]
        # Sort by distance
        distances.sort(key=lambda tup: tup[0])
        # Determine effective k (cannot exceed number of training points)
        effective_k = min(self.k, len(distances))
        # Majority vote among the k nearest neighbours
        neighbour_labels = [label for _, label in distances[:effective_k]]
        return self._majority_vote(neighbour_labels)

    @staticmethod
    def _majority_vote(labels: List[Any]) -> Any:
        """Return the most common label; break ties by deterministic ordering."""
        if not labels:
            raise ValueError("No labels to vote on")
        frequency: Dict[Any, int] = {}
        for lbl in labels:
            frequency[lbl] = frequency.get(lbl, 0) + 1
        # Find max frequency
        max_freq = max(frequency.values())
        # Gather all labels with max frequency and return the smallest (deterministic)
        candidates = [lbl for lbl, cnt in frequency.items() if cnt == max_freq]
        return min(candidates)  # type: ignore[arg-type]


# End of core.py
# ---------------------------------------------------------------------------
