import unittest
import random
import time
from typing import List
from core import PathIntegralSurrogate, dot, l2_norm


def _random_vector(dim: int, low: float = -1.0, high: float = 1.0) -> List[float]:
    return [random.uniform(low, high) for _ in range(dim)]


class TestPathIntegralSurrogate(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(0)
        self.dim = 5
        self.steps = 8
        self.lr = 0.1
        self.inverter = PathIntegralSurrogate(learning_rate=self.lr, steps=self.steps)
        self.gradients = [_random_vector(self.dim) for _ in range(self.steps)]
        for g in self.gradients:
            self.inverter.add_step(g)

    def test_compute_surrogate_matches_manual(self) -> None:
        surrogate = self.inverter.compute_surrogate()
        # Manual computation
        manual = [0.0] * self.dim
        for t, g in enumerate(self.gradients):
            weight = 1.0 - t / self.steps
            for i, val in enumerate(g):
                manual[i] += self.lr * weight * val
        self.assertTrue(all(abs(a - b) < 1e-12 for a, b in zip(surrogate, manual)))

    def test_invert_is_consistent(self) -> None:
        surrogate = self.inverter.compute_surrogate()
        approx_original = self.inverter.invert(surrogate)
        # The inversion should recover a vector proportional to the average of gradients
        avg_grad = [sum(g[i] for g in self.gradients) / self.steps for i in range(self.dim)]
        # Compare direction (cosine similarity)
        dot_prod = dot(approx_original, avg_grad)
        norm_prod = l2_norm(approx_original) * l2_norm(avg_grad)
        cosine = dot_prod / norm_prod if norm_prod else 0.0
        self.assertGreater(cosine, 0.99)  # vectors should be almost aligned

    def test_reset_clears_state(self) -> None:
        self.inverter.reset()
        self.assertRaises(RuntimeError, self.inverter.compute_surrogate)
        # Reuse after reset
        for _ in range(self.steps):
            self.inverter.add_step(_random_vector(self.dim))
        self.assertIsInstance(self.inverter.compute_surrogate(), list)


def benchmark_surrogate(steps: int = 1000, dim: int = 50, repeats: int = 5) -> None:
    """Simple benchmark printing average runtime."""
    lr = 0.01
    inverter = PathIntegralSurrogate(lr, steps)
    for _ in range(steps):
        inverter.add_step(_random_vector(dim))
    start = time.perf_counter()
    for _ in range(repeats):
        inverter.compute_surrogate()
    elapsed = time.perf_counter() - start
    print(f"Benchmark: {steps} steps, dim={dim}, repeats={repeats}")
    print(f"  Avg compute time: {elapsed / repeats * 1e3:.3f} ms")


if __name__ == '__main__':
    # Run unit tests
    unittest.main(exit=False)

    # Demonstration
    print("\n--- Demo: Path Integral Surrogate ---")
    demo_steps = 6
    demo_dim = 3
    demo_lr = 0.05
    demo_inv = PathIntegralSurrogate(demo_lr, demo_steps)
    print("Generating random gradients:")
    for i in range(demo_steps):
        g = _random_vector(demo_dim)
        demo_inv.add_step(g)
        print(f"  step {i}: {g}")
    surrogate = demo_inv.compute_surrogate()
    print(f"\nSurrogate (weighted sum): {surrogate}")
    approx = demo_inv.invert(surrogate)
    print(f"Approximate original gradient (scaled back): {approx}")

    # Run a quick benchmark
    benchmark_surrogate(steps=2000, dim=100, repeats=3)
