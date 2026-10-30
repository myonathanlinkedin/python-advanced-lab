import unittest
from typing import List

# Import core components
from core import KernelConfig, Agent, OptimizerLoop, default_fitness


class TestKernelConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(42)

    def test_mutation_respects_rate(self) -> None:
        cfg = KernelConfig((5, 5, 5), max_param=10)
        mutated = cfg.mutate(0.0, self.rng)
        self.assertEqual(mutated, cfg, "Zero mutation rate must leave config unchanged")
        mutated = cfg.mutate(1.0, self.rng)
        # All params should be random now, but still within bounds
        for p in mutated.params:
            self.assertTrue(0 <= p <= cfg.max_param)

    def test_crossover_preserves_length(self) -> None:
        a = KernelConfig((1, 2, 3, 4), max_param=10)
        b = KernelConfig((4, 3, 2, 1), max_param=10)
        child = KernelConfig.crossover(a, b, self.rng)
        self.assertEqual(len(child.params), len(a.params))
        # Child must consist of a prefix from a and suffix from b
        # Find crossover point by comparing with parents
        diff = [i for i, (pa, pc) in enumerate(zip(a.params, child.params)) if pa != pc]
        if diff:
            point = diff[0]
            self.assertEqual(child.params[:point], a.params[:point])
            self.assertEqual(child.params[point:], b.params[point:])

    def test_invalid_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            KernelConfig((-1, 0, 1), max_param=5)
        with self.assertRaises(ValueError):
            KernelConfig((0, 6), max_param=5)


class TestAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.best = KernelConfig((2, 2, 2), max_param=5)
        self.population = [
            KernelConfig((0, 0, 0), max_param=5),
            KernelConfig((5, 5, 5), max_param=5)
        ]
        self.agent = Agent(seed=123, mutation_rate=0.5, crossover_rate=0.5)

    def test_propose_mutation(self) -> None:
        # Force mutation path by setting crossover_rate to 0
        self.agent.crossover_rate = 0.0
        proposal = self.agent.propose(self.best, self.population)
        # Proposal must be a mutation of best, not identical unless RNG hits 0
        self.assertNotEqual(proposal, self.best)

    def test_propose_crossover(self) -> None:
        self.agent.crossover_rate = 1.0
        proposal = self.agent.propose(self.best, self.population)
        # Proposal must be derived from crossover then mutation
        self.assertIsInstance(proposal, KernelConfig)

    def test_empty_population_error(self) -> None:
        with self.assertRaises(ValueError):
            self.agent.propose(self.best, [])


class TestOptimizerLoop(unittest.TestCase):
    def test_initial_best_is_max_fitness(self) -> None:
        loop = OptimizerLoop(num_agents=4, config_dim=3, max_param=5, seed=0)
        fitnesses = [default_fitness(cfg) for cfg in loop.population]
        self.assertEqual(default_fitness(loop.best), max(fitnesses))

    def test_step_improves_or_keeps_best(self) -> None:
        loop = OptimizerLoop(num_agents=5, config_dim=4, max_param=7, seed=1)
        initial_best_fit = default_fitness(loop.best)
        loop.step()
        self.assertGreaterEqual(default_fitness(loop.best), initial_best_fit)

    def test_run_returns_best_after_iterations(self) -> None:
        loop = OptimizerLoop(num_agents=6, config_dim=5, max_param=9, seed=42)
        best = loop.run(iterations=50)
        # Verify that best is indeed the max of the final population
        self.assertEqual(best, max(loop.population, key=default_fitness))

    def test_invalid_initialisation(self) -> None:
        with self.assertRaises(ValueError):
            OptimizerLoop(num_agents=0, config_dim=3)
        with self.assertRaises(ValueError):
            OptimizerLoop(num_agents=3, config_dim=0)

    def test_negative_iterations(self) -> None:
        loop = OptimizerLoop(num_agents=3, config_dim=2)
        with self.assertRaises(ValueError):
            loop.run(-5)


def benchmark_optimisation(iterations: int = 200) -> float:
    """
    Simple benchmark returning the fitness of the best configuration after
    `iterations` steps. Deterministic because of fixed seed.
    """
    loop = OptimizerLoop(num_agents=8, config_dim=6, max_param=10, seed=12345)
    best = loop.run(iterations)
    return default_fitness(best)


if __name__ == '__main__':
    # Run unit tests
    unittest.main(exit=False)

    # Demo benchmark
    final_fitness = benchmark_optimisation(200)
    print(f"Benchmark completed. Final best fitness: {final_fitness:.2f}")
