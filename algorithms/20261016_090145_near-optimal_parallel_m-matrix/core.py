"""
Near-Optimal Parallel M‑Matrix Scaling (reference implementation)

The algorithm scales a non‑negative square matrix A (interpreted as an
M‑matrix with positive diagonal) to a matrix B = D_r · A · D_c whose
row and column sums are (approximately) 1.  The method is the classic
Sinkhorn–Knopp iteration, which is provably convergent for matrices that
are *scalable* (i.e. have total support).  For M‑matrices this condition
holds, and the iteration is embarrassingly parallel: each iteration
requires only independent row/column reductions.

Complexities
------------
*Time*   : O(k·n²) where n is the dimension of A and k the number of
           iterations until convergence.
*Space*  : O(n²) for the matrix plus O(n) for the scaling vectors.

The implementation is deterministic, uses only the Python standard
library, and raises clear exceptions for malformed inputs.
"""

from __future__ import annotations

import math
from typing import List, Tuple


class MMatrixScaler:
    """
    Scales a square non‑negative matrix to (approximately) doubly stochastic
    form using the Sinkhorn–Knopp iteration.

    Parameters
    ----------
    max_iter : int, optional
        Upper bound on the number of iterations (default 10_000).
    tol : float, optional
        Desired absolute tolerance on the maximum deviation of any row or
        column sum from 1 (default 1e-6).
    """

    def __init__(self, max_iter: int = 10_000, tol: float = 1e-6) -> None:
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _validate_matrix(A: List[List[float]]) -> int:
        """Validate that A is a square matrix with non‑negative entries and
        a strictly positive diagonal.  Returns the dimension n."""
        if not isinstance(A, list) or not A:
            raise ValueError("A must be a non‑empty list of lists")
        n = len(A)
        for i, row in enumerate(A):
            if not isinstance(row, list) or len(row) != n:
                raise ValueError("A must be square (all rows length n)")
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise TypeError("Matrix entries must be numeric")
                if val < 0:
                    raise ValueError("Matrix entries must be non‑negative")
                if i == j and val <= 0:
                    raise ValueError("Diagonal entries must be strictly positive")
        return n

    def scale(
        self, A: List[List[float]]
    ) -> Tuple[List[List[float]], List[float], List[float]]:
        """
        Perform the scaling.

        Returns
        -------
        B : List[List[float]]
            The scaled matrix D_r·A·D_c.
        r : List[float]
            Row scaling factors (diagonal of D_r).
        c : List[float]
            Column scaling factors (diagonal of D_c).

        Raises
        ------
        ValueError
            If A is not a valid M‑matrix for this routine.
        """
        n = self._validate_matrix(A)

        # Convert to mutable float matrix (deep copy) to avoid side effects.
        A_float: List[List[float]] = [[float(A[i][j]) for j in range(n)] for i in range(n)]

        # Initialise scaling vectors.
        r: List[float] = [1.0] * n
        c: List[float] = [1.0] * n

        for iteration in range(self.max_iter):
            # ---- Row scaling step ----
            row_sums = [
                sum(A_float[i][j] * c[j] for j in range(n)) for i in range(n)
            ]
            # Guard against division by zero (should not happen for scalable matrices).
            r = [1.0 / rs if rs > 0 else 0.0 for rs in row_sums]

            # ---- Column scaling step ----
            col_sums = [
                sum(r[i] * A_float[i][j] for i in range(n)) for j in range(n)
            ]
            c = [1.0 / cs if cs > 0 else 0.0 for cs in col_sums]

            # ---- Convergence check ----
            # Compute the scaled matrix B = diag(r)·A·diag(c)
            B = [
                [r[i] * A_float[i][j] * c[j] for j in range(n)] for i in range(n)
            ]

            max_row_err = max(
                abs(sum(B[i][j] for j in range(n)) - 1.0) for i in range(n)
            )
            max_col_err = max(
                abs(sum(B[i][j] for i in range(n)) - 1.0) for j in range(n)
            )
            if max(max_row_err, max_col_err) < self.tol:
                # Converged
                break
        else:
            # Loop exhausted without break
            raise RuntimeError(
                f"Scaling did not converge within {self.max_iter} iterations"
            )

        return B, r, c
