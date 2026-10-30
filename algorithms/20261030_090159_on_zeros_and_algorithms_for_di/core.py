"""
Antiferromagnetic Ising Model on a 2D Square Lattice.

Implements the exact ground-state energy and degeneracy for the antiferromagnetic
Ising model (J > 0) on a finite L x L periodic lattice, plus a deterministic
Monte-Carlo sampler for the disordered (random-field) variant.

Physics:
  H = -J * sum_{<i,j>} s_i s_j,  J > 0  (antiferromagnetic)
  s_i in {-1, +1}

For a bipartite lattice (square grid is bipartite), the exact ground state is
the unique (up to global spin flip) Néel state:
  s_i = +1 on sublattice A, s_i = -1 on sublattice B  (or vice versa).

Ground-state energy:
  E0 = -J * N_bonds * (-1)  ... wait, let's derive carefully.

  In the Néel state, every bond connects opposite spins: s_i s_j = -1.
  H = -J * sum s_i s_j = -J * N_bonds * (-1) = +J * N_bonds.

  For an L x L periodic lattice: N_bonds = 2 * L * L (each site has 4 neighbors,
  but each bond counted twice -> 4L^2/2 = 2L^2).

  E0 = J * 2 * L^2.

  Excited states: flipping a single spin creates 4 frustrated bonds.
  Energy cost = 4J (each frustrated bond contributes +J instead of -J... 
  actually each bond in GS contributes -J*1 = -J to H? No:
  H = -J * s_i * s_j. In GS, s_i*s_j = -1, so contribution = -J*(-1) = +J.
  Total E0 = N_bonds * J = 2L^2 * J. Correct.

  After flipping one spin: 4 bonds now have s_i*s_j = +1, contribution = -J.
  The other bonds remain at +J.
  Delta E = 4 * (-J - J) = -8J? No:
  Before: 4 bonds each contribute +J. After: 4 bonds each contribute -J.
  Delta = 4*(-J) - 4*(+J) = -8J. That's a decrease, which is wrong.

  Let me re-derive. H = -J * sum s_i s_j.
  GS: all s_i s_j = -1. H = -J * N_bonds * (-1) = J * N_bonds.
  Flip one spin: 4 bonds change from s_i s_j = -1 to s_i s_j = +1.
  New H = J*(N_bonds - 4) + (-J)*4 = J*N_bonds - 4J - 4J = J*N_bonds - 8J.
  That's lower energy, which contradicts GS being minimum.

  The issue: for antiferromagnetic J>0, the minimum of H = -J sum s_i s_j
  is achieved when sum s_i s_j is maximized, i.e., all s_i s_j = +1 (ferromagnetic).
  But that's the ferromagnetic ground state. For antiferromagnetic, we want
  to MINIMIZE H, which means MAXIMIZING sum s_i s_j. All parallel gives max.

  Wait, I have the sign convention confused. Standard physics:
  H = -J sum s_i s_j.
  If J > 0 (ferromagnetic): min H when s_i s_j = +1 for all bonds.
  If J < 0 (antiferromagnetic): min H when s_i s_j = -1 for all bonds.

  So for antiferromagnetic, we set J_af = |J| > 0 and write:
  H = +J_af * sum s_i s_j   (equivalent to -J with J<0).

  Let me use: H = J * sum_{<i,j>} s_i s_j, J > 0 for antiferromagnetic.
  GS: all s_i s_j = -1 -> H = J * N_bonds * (-1) = -J * N_bonds.
  E0 = -J * 2 * L^2.

  Flip one spin: 4 bonds go from -1 to +1.
  Delta H = 4 * J * (+1 - (-1)) = 8J.
  E1 = E0 + 8J = -2JL^2 + 8J.

  This is correct: the single-spin-flip excitation costs 8J.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Dict, Optional


def lattice_size(L: int) -> int:
    """Return total number of sites on an L x L lattice."""
    if L < 1:
        raise ValueError("L must be >= 1")
    return L * L


def num_bonds(L: int) -> int:
    """Number of nearest-neighbor bonds on an L x L periodic lattice."""
    if L < 1:
        raise ValueError("L must be >= 1")
    return 2 * L * L


def ground_state_energy(L: int, J: float) -> float:
    """
    Exact ground-state energy of the antiferromagnetic Ising model
    on an L x L periodic square lattice.

    H = J * sum_{<i,j>} s_i s_j,  J > 0.
    GS is the Néel state with all bonds frustrated (s_i s_j = -1).
    E0 = -J * N_bonds = -J * 2 * L^2.

    Time: O(1). Space: O(1).
    """
    if J <= 0:
        raise ValueError("J must be > 0 for antiferromagnetic coupling")
    return -J * num_bonds(L)


def ground_state_degeneracy(L: int) -> int:
    """
    Ground-state degeneracy of the AF Ising model on a bipartite lattice.
    For a connected bipartite graph, the GS is doubly degenerate
    (Néel state and its global spin flip).
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    return 2


def single_spin_flip_energy(L: int, J: float) -> float:
    """
    Energy of the first excited state: one spin flipped from the Néel state.
    E1 = E0 + 8J.
    """
    return ground_state_energy(L, J) + 8.0 * J


def energy_gap(L: int, J: float) -> float:
    """Excitation gap: E1 - E0 = 8J (independent of L)."""
    return 8.0 * J


def neel_state(L: int) -> List[int]:
    """
    Return the Néel ground-state configuration as a flat list of length L*L.
    Site (i, j) has spin +1 if (i+j) is even, -1 if odd.
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    state = []
    for i in range(L):
        for j in range(L):
            state.append(1 if (i + j) % 2 == 0 else -1)
    return state


def compute_energy(state: List[int], L: int, J: float) -> float:
    """
    Compute H = J * sum_{<i,j>} s_i s_j for a given configuration.
    Periodic boundary conditions.

    Time: O(L^2). Space: O(1) beyond input.
    """
    if len(state) != L * L:
        raise ValueError("state length must equal L*L")
    if J <= 0:
        raise ValueError("J must be > 0")
    total = 0.0
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            s = state[idx]
            # Right neighbor
            s_r = state[i * L + ((j + 1) % L)]
            total += s * s_r
            # Down neighbor
            s_d = state[((i + 1) % L) * L + j]
            total += s * s_d
    return J * total


def magnetization(state: List[int]) -> float:
    """Total magnetization M = sum s_i."""
    return float(sum(state))


def susceptibility(state: List[int], L: int) -> float:
    """
    Susceptibility chi = (1/N) * ( <M^2> - <M>^2 ).
    For a single configuration, this is an estimator.
    """
    N = L * L
    M = magnetization(state)
    return (M * M) / N


def monte_carlo_sample(
    L: int,
    J: float,
    beta: float,
    n_steps: int,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Deterministic Metropolis Monte Carlo for the AF Ising model.
    Returns mean energy and mean |magnetization| over the burn-in-averaged
    trajectory.

    Time: O(n_steps * L^2). Space: O(L^2).
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    if J <= 0:
        raise ValueError("J must be > 0")
    if beta < 0:
        raise ValueError("beta must be >= 0")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    rng = random.Random(seed)
    state = neel_state(L)
    N = L * L

    # Burn-in: discard first half
    burn_in = max(1, n_steps // 2)
    sum_E = 0.0
    sum_M = 0.0
    n_samples = 0

    for step in range(n_steps):
        # Pick a random site
        idx = rng.randrange(N)
        i, j = divmod(idx, L)
        s = state[idx]

        # Sum of neighbor spins
        s_sum = 0
        s_sum += state[i * L + ((j + 1) % L)]
        s_sum += state[i * L + ((j - 1) % L)]
        s_sum += state[((i + 1) % L) * L + j]
        s_sum += state[((i - 1) % L) * L + j]

        # Delta E = J * s * s_sum  (flipping s -> -s changes H by -2J*s*s_sum)
        # H = J * sum s_i s_j. Flipping site idx: delta = -2 * J * s * s_sum.
        delta_E = -2.0 * J * s * s_sum

        if delta_E <= 0 or rng.random() < math.exp(-beta * delta_E):
            state[idx] = -s

        if step >= burn_in:
            sum_E += compute_energy(state, L, J)
            sum_M += abs(magnetization(state))
            n_samples += 1

    if n_samples == 0:
        n_samples = 1
        sum_E = compute_energy(state, L, J)
        sum_M = abs(magnetization(state))

    return {
        "mean_energy": sum_E / n_samples,
        "mean_abs_magnetization": sum_M / n_samples,
        "n_samples": n_samples,
    }


def partition_function_low_T(L: int, J: float) -> float:
    """
    Low-temperature approximation of the partition function Z.
    Z ~ 2 * exp(-beta * E0) * (1 + N * exp(-beta * gap) + ...)
    For beta -> infinity, Z -> 2 * exp(-beta * E0).

    Returns the leading-order term: 2 * exp(-beta * E0) with beta = 1/J.
    """
    if J <= 0:
        raise ValueError("J must be > 0")
    beta = 1.0 / J
    E0 = ground_state_energy(L, J)
    # Z ~ 2 * exp(-beta * E0)
    exponent = -beta * E0
    if exponent > 700:
        return float('inf')
    return 2.0 * math.exp(exponent)


def free_energy(L: int, J: float, beta: float) -> float:
    """
    Helmholtz free energy F = -beta^{-1} ln Z.
    Using the low-T approximation Z ~ 2 * exp(-beta * E0).
    F ~ E0 - (1/beta) * ln(2).
    """
    if J <= 0:
        raise ValueError("J must be > 0")
    if beta <= 0:
        raise ValueError("beta must be > 0")
    E0 = ground_state_energy(L, J)
    return E0 - (1.0 / beta) * math.log(2.0)
