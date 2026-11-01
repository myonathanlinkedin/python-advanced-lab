"""
core.py
---------

Implementation of a linear consensus dynamics algorithm with optional
node sampling at each iteration. The weight matrix is constructed using
the Metropolis-Hastings rule, guaranteeing that it is doubly stochastic
and thus the dynamics converge to the average of the initial states on
any connected undirected graph.

The module provides:
- Graph data structure with helpers to build common topologies.
- Functions to compute the Metropolis weight matrix.
- Functions to perform one consensus step with a given sample set.
- A driver to run the dynamics for a fixed number of steps.
- Utility functions to assess convergence.
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Callable, Iterable


class Graph:
    """Undirected weighted graph represented by a symmetric adjacency matrix."""

    def __init__(self, adjacency: List[List[float]]) -> None:
        n = len(adjacency)
        if any(len(row) != n for row in adjacency):
            raise ValueError("Adjacency matrix must be square.")
        self.n: int = n
        # Ensure symmetry and non‑negative weights
        for i in range(n):
            for j in range(n):
                if adjacency[i][j] != adjacency[j][i]:
                    raise ValueError("Adjacency matrix must be symmetric.")
                if adjacency[i][j] < 0:
                    raise ValueError("Edge weights must be non‑negative.")
        self.adj: List[List[float]] = [row[:] for row in adjacency]

    @property
    def degree(self) -> List[float]:
        """Degree of each node (sum of incident edge weights)."""
        return [sum(self.adj[i]) for i in range(self.n)]

    @staticmethod
    def complete(n: int, weight: float = 1.0) -> "Graph":
        """Return a complete graph on n nodes with uniform edge weight."""
        if n <= 0:
            raise ValueError("Number of nodes must be positive.")
        adj = [[weight if i != j else 0.0 for j in range(n)] for i in range(n)]
        return Graph(adj)

    @staticmethod
    def path(n: int, weight: float = 1.0) -> "Graph":
        """Return a simple path (line) graph on n nodes."""
        if n <= 0:
            raise ValueError("Number of nodes must be positive.")
        adj = [[0.0] * n for _ in range(n)]
        for i in range(n - 1):
            adj[i][i + 1] = adj[i + 1][i] = weight
        return Graph(adj)


def metropolis_weights(g: Graph) -> List[List[float]]:
    """
    Construct the Metropolis weight matrix W for graph g.
    W is doubly stochastic, symmetric, and respects the graph topology.
    """
    n = g.n
    deg = g.degree
    W = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j and g.adj[i][j] > 0:
                W[i][j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        # Diagonal entry ensures rows sum to 1
        W[i][i] = 1.0 - sum(W[i][j] for j in range(n) if j != i)
    return W


def consensus_step(state: List[float],
                   W: List[List[float]],
                   sample: Iterable[int]) -> List[float]:
    """
    Perform a single consensus iteration.
    Only nodes whose index appears in `sample` are updated; others retain
    their previous value.
    """
    n = len(state)
    new_state = state[:]
    sample_set = set(sample)
    for i in sample_set:
        acc = 0.0
        row = W[i]
        for j in range(n):
            acc += row[j] * state[j]
        new_state[i] = acc
    return new_state


def run_consensus(initial: List[float],
                  W: List[List[float]],
                  steps: int,
                  sample_rate: float,
                  rng: random.Random) -> List[List[float]]:
    """
    Execute the consensus dynamics for a given number of steps.

    Parameters
    ----------
    initial : list of float
        Starting values for each node.
    W : weight matrix (n×n)
    steps : int
        Number of iterations to perform.
    sample_rate : float in (0,1]
        Probability that a node is selected for update at each step.
    rng : random.Random
        Random generator for deterministic sampling.

    Returns
    -------
    history : list of state vectors (length steps+1)
    """
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError("sample_rate must be in (0, 1].")
    n = len(initial)
    history: List[List[float]] = [initial[:]]
    state = initial[:]

    for _ in range(steps):
        # Determine which nodes update this round
        sample = [i for i in range(n) if rng.random() < sample_rate]
        # If sample is empty (possible for very low rates), fall back to full update
        if not sample:
            sample = list(range(n))
        state = consensus_step(state, W, sample)
        history.append(state[:])
    return history


def max_deviation(state: List[float]) -> float:
    """Maximum absolute deviation from the average (consensus error)."""
    if not state:
        return 0.0
    avg = sum(state) / len(state)
    return max(abs(x - avg) for x in state)


def is_converged(state: List[float], tol: float = 1e-8) -> bool:
    """Return True if all values are within `tol` of each other."""
    return max_deviation(state) <= tol
