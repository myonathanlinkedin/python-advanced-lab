"""
main.py
---------

Unit‑test suite for the consensus dynamics implementation.
The tests cover:
- Convergence on complete and path graphs with full updates.
- Effect of stochastic sampling on convergence speed.
- Monotonicity of the consensus error for deterministic updates.
"""

import unittest
import random
from typing import List

# Import from core.py (assumes same directory)
from core import (
    Graph,
    metropolis_weights,
    run_consensus,
    max_deviation,
    is_converged,
)


class TestConsensusDynamics(unittest.TestCase):
    def setUp(self) -> None:
        # Fixed seed for reproducibility
        self.rng = random.Random(42)

    def _run_and_check(self,
                       graph: Graph,
                       steps: int,
                       sample_rate: float,
                       tol: float) -> List[List[float]]:
        n = graph.n
        # Random initial values in [0, 10)
        init = [self.rng.uniform(0, 10) for _ in range(n)]
        W = metropolis_weights(graph)
        history = run_consensus(init, W, steps, sample_rate, self.rng)
        final_state = history[-1]
        self.assertTrue(is_converged(final_state, tol),
                        msg=f"Did not converge within tolerance {tol}")
        return history

    def test_complete_graph_full_update(self) -> None:
        """Complete graph should converge rapidly with full updates."""
        g = Graph.complete(5)
        history = self._run_and_check(g, steps=50, sample_rate=1.0, tol=1e-6)
        # Verify monotonic decrease of max deviation
        deviations = [max_deviation(s) for s in history]
        for earlier, later in zip(deviations, deviations[1:]):
            self.assertLessEqual(later, earlier + 1e-12)

    def test_path_graph_full_update(self) -> None:
        """Path graph converges slower but still reaches consensus."""
        g = Graph.path(7)
        self._run_and_check(g, steps=200, sample_rate=1.0, tol=1e-5)

    def test_sampling_rate_effect(self) -> None:
        """Even with 50 % sampling the system should converge given enough steps."""
        g = Graph.complete(6)
        # Use a relatively low sample rate; increase steps to compensate
        self._run_and_check(g, steps=500, sample_rate=0.5, tol=2e-5)

    def test_single_node_trivial(self) -> None:
        """A graph with a single node is already at consensus."""
        g = Graph.complete(1)
        history = self._run_and_check(g, steps=1, sample_rate=1.0, tol=0.0)
        self.assertEqual(len(history), 2)  # initial + one step
        self.assertAlmostEqual(history[-1][0], history[0][0])

    def test_invalid_parameters(self) -> None:
        """Check that invalid arguments raise appropriate errors."""
        g = Graph.complete(3)
        W = metropolis_weights(g)
        init = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            run_consensus(init, W, steps=10, sample_rate=0.0, rng=self.rng)
        with self.assertRaises(ValueError):
            run_consensus(init, W, steps=10, sample_rate=1.2, rng=self.rng)


if __name__ == "__main__":
    unittest.main()
