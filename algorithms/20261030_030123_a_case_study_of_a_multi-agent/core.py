import random
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional


@dataclass(frozen=True)
class KernelConfig:
    """
    Immutable representation of a kernel configuration.
    Parameters are integer values in a fixed range [0, max_param].
    """
    params: Tuple[int, ...]
    max_param: int = field(default=10, compare=False)

    def __post_init__(self) -> None:
        if any(p < 0 or p > self.max_param for p in self.params):
            raise ValueError("All parameters must be within [0, max_param]")

    def mutate(self, mutation_rate: float, rng: random.Random) -> "KernelConfig":
        """
        Return a new KernelConfig where each parameter has a chance
        `mutation_rate` to be replaced by a random value.
        """
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be between 0 and 1")
        new_params = list(self.params)
        for i in range(len(new_params)):
            if rng.random() < mutation_rate:
                new_params[i] = rng.randint(0, self.max_param)
        return KernelConfig(tuple(new_params), self.max_param)

    @staticmethod
    def crossover(parent_a: "KernelConfig", parent_b: "KernelConfig",
                  rng: random.Random) -> "KernelConfig":
        """
        Single-point crossover between two parents.
        """
        if len(parent_a.params) != len(parent_b.params):
            raise ValueError("Parents must have the same number of parameters")
        point = rng.randint(1, len(parent_a.params) - 1)
        child_params = parent_a.params[:point] + parent_b.params[point:]
        return KernelConfig(child_params, parent_a.max_param)


def default_fitness(config: KernelConfig) -> float:
    """
    Deterministic fitness function used for the case study.
    Higher is better. Example: weighted sum of parameters where
    later parameters have higher weight.
    """
    weight = 1
    total = 0.0
    for p in config.params:
        total += weight * p
        weight *= 2  # exponential weighting
    return total


class Agent:
    """
    An agent that proposes new kernel configurations based on the current best.
    It keeps an internal random generator for reproducibility.
    """
    def __init__(self,
                 seed: int,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.5):
        self.rng = random.Random(seed)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def propose(self,
                best: KernelConfig,
                population: List[KernelConfig]) -> KernelConfig:
        """
        Propose a new configuration.
        With probability `crossover_rate` perform crossover with a random peer,
        otherwise mutate the best configuration.
        """
        if not population:
            raise ValueError("Population must contain at least one configuration")
        if self.rng.random() < self.crossover_rate:
            peer = self.rng.choice(population)
            child = KernelConfig.crossover(best, peer, self.rng)
            return child.mutate(self.mutation_rate, self.rng)
        else:
            return best.mutate(self.mutation_rate, self.rng)


class OptimizerLoop:
    """
    Core loop that coordinates multiple agents to discover an optimal kernel.
    """
    def __init__(self,
                 num_agents: int,
                 config_dim: int,
                 max_param: int = 10,
                 fitness_fn: Callable[[KernelConfig], float] = default_fitness,
                 seed: int = 0):
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if config_dim <= 0:
            raise ValueError("config_dim must be positive")
        self.rng = random.Random(seed)
        self.fitness_fn = fitness_fn
        self.agents: List[Agent] = [
            Agent(self.rng.randint(0, 2**31 - 1)) for _ in range(num_agents)
        ]
        # Initialise a random population
        self.population: List[KernelConfig] = [
            KernelConfig(
                tuple(self.rng.randint(0, max_param) for _ in range(config_dim)),
                max_param
            )
            for _ in range(num_agents)
        ]
        self.best: KernelConfig = max(self.population,
                                      key=self.fitness_fn)

    def step(self) -> None:
        """
        Perform a single optimization iteration.
        Each agent proposes a candidate; the best candidate replaces the worst
        member of the population if it improves fitness.
        """
        proposals: List[KernelConfig] = [
            agent.propose(self.best, self.population) for agent in self.agents
        ]
        # Evaluate proposals
        scored = [(cfg, self.fitness_fn(cfg)) for cfg in proposals]
        # Sort by fitness descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Replace worst individuals if better
        for cfg, fit in scored:
            worst_cfg = min(self.population, key=self.fitness_fn)
            worst_fit = self.fitness_fn(worst_cfg)
            if fit > worst_fit:
                self.population.remove(worst_cfg)
                self.population.append(cfg)
                if fit > self.fitness_fn(self.best):
                    self.best = cfg
            else:
                # Since proposals are sorted, remaining ones cannot improve
                break

    def run(self, iterations: int) -> KernelConfig:
        """
        Execute the optimization loop for a fixed number of iterations.
        Returns the best configuration found.
        """
        if iterations < 0:
            raise ValueError("iterations must be non‑negative")
        for _ in range(iterations):
            self.step()
        return self.best

    def get_population_fitness(self) -> List[float]:
        """Utility for testing/analysis: fitness of each member."""
        return [self.fitness_fn(cfg) for cfg in self.population]
