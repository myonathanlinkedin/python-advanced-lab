"""pendulum_clock.py
A self‑contained module for modeling a simple pendulum and a 30‑minute trash clock.
Provides physics calculations, simulation utilities, and a minimal test suite.

The module is deliberately lightweight, uses only the Python standard library,
and follows production‑grade practices: type hints, comprehensive docstrings,
input validation, and deterministic behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


class PendulumError(ValueError):
    """Custom exception for invalid pendulum parameters."""


@dataclass(frozen=True, slots=True)
class Pendulum:
    """
    Simple pendulum model using the small‑angle approximation.

    Attributes
    ----------
    length_m : float
        Length of the pendulum rod in meters. Must be positive.
    gravity_m_s2 : float, default 9.80665
        Gravitational acceleration in meters per second squared. Must be positive.
    """

    length_m: float
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        if self.length_m <= 0:
            raise PendulumError("Pendulum length must be positive.")
        if self.gravity_m_s2 <= 0:
            raise PendulumError("Gravity must be positive.")

    @property
    def angular_frequency(self) -> float:
        """
        Angular frequency ω = sqrt(g / L) in radians per second.

        Returns
        -------
        float
            Angular frequency.
        """
        return math.sqrt(self.gravity_m_s2 / self.length_m)

    def period(self) -> float:
        """
        Compute the period T = 2π / ω.

        Returns
        -------
        float
            Period of the pendulum in seconds.
        """
        return 2.0 * math.pi / self.angular_frequency

    @staticmethod
    def length_for_period(target_period_s: float, gravity_m_s2: float = 9.80665) -> float:
        """
        Compute the required pendulum length for a desired period using
        T = 2π sqrt(L / g)  ⇒  L = g (T / 2π)^2.

        Parameters
        ----------
        target_period_s : float
            Desired period in seconds. Must be positive.
        gravity_m_s2 : float, optional
            Gravitational acceleration. Must be positive.

        Returns
        -------
        float
            Required length in meters.

        Raises
        ------
        PendulumError
            If any argument is non‑positive.
        """
        if target_period_s <= 0:
            raise PendulumError("Target period must be positive.")
        if gravity_m_s2 <= 0:
            raise PendulumError("Gravity must be positive.")
        return gravity_m_s2 * (target_period_s / (2.0 * math.pi)) ** 2

    def angle_at_time(self, t: float, initial_angle_rad: float) -> float:
        """
        Compute the angular displacement at time t using simple harmonic motion:
        θ(t) = θ₀ cos(ω t).

        Parameters
        ----------
        t : float
            Time elapsed since start, in seconds. Must be non‑negative.
        initial_angle_rad : float
            Initial angular displacement in radians. Small‑angle approximation assumed.

        Returns
        -------
        float
            Angular displacement at time t (radians).

        Raises
        ------
        PendulumError
            If t is negative.
        """
        if t < 0:
            raise PendulumError("Time cannot be negative.")
        return initial_angle_rad * math.cos(self.angular_frequency * t)

    def simulate(
        self,
        duration_s: float,
        dt_s: float,
        initial_angle_rad: float = 0.1,
    ) -> List[Tuple[float, float]]:
        """
        Simulate pendulum motion over a given duration.

        Parameters
        ----------
        duration_s : float
            Total simulation time in seconds. Must be positive.
        dt_s : float
            Time step for the simulation in seconds. Must be positive and <= duration.
        initial_angle_rad : float, optional
            Starting angle (radians). Default 0.1 rad (~5.7°).

        Returns
        -------
        List[Tuple[float, float]]
            List of (time, angle) tuples.

        Raises
        ------
        PendulumError
            If any argument is invalid.
        """
        if duration_s <= 0:
            raise PendulumError("Duration must be positive.")
        if dt_s <= 0:
            raise PendulumError("Time step must be positive.")
        if dt_s > duration_s:
            raise PendulumError("Time step cannot exceed duration.")
        if not isinstance(initial_angle_rad, (int, float)):
            raise PendulumError("Initial angle must be a number.")

        steps = int(math.ceil(duration_s / dt_s)) + 1
        result: List[Tuple[float, float]] = []
        for i in range(steps):
            t = min(i * dt_s, duration_s)
            angle = self.angle_at_time(t, initial_angle_rad)
            result.append((t, angle))
        return result


class TrashClock:
    """
    Minimal clock model driven by a pendulum. The clock ticks each time the pendulum
    reaches an extremum (i.e., every half period).

    Attributes
    ----------
    pendulum : Pendulum
        The underlying pendulum driving the clock.
    """

    def __init__(self, pendulum: Pendulum) -> None:
        if not isinstance(pendulum, Pendulum):
            raise TypeError("pendulum must be an instance of Pendulum.")
        self.pendulum = pendulum
        self._half_period = self.pendulum.period() / 2.0

    @property
    def tick_interval(self) -> float:
        """
        Time between successive ticks (seconds).

        Returns
        -------
        float
            Half the pendulum period.
        """
        return self._half_period

    def next_tick(self, current_time_s: float) -> float:
        """
        Compute the next tick time after the given current time.

        Parameters
        ----------
        current_time_s : float
            Current time in seconds. Must be non‑negative.

        Returns
        -------
        float
            Timestamp of the next tick.

        Raises
        ------
        ValueError
            If current_time_s is negative.
        """
        if current_time_s < 0:
            raise ValueError("Current time cannot be negative.")
        # Number of half‑periods elapsed (ceil to next integer)
        periods_elapsed = math.ceil(current_time_s / self._half_period)
        return periods_elapsed * self._half_period

    def __repr__(self) -> str:
        return f"TrashClock(pendulum=Pendulum(length_m={self.pendulum.length_m:.3f}, gravity_m_s2={self.pendulum.gravity_m_s2:.3f}))"


def _demo() -> None:
    """Demonstrate a 30‑minute pendulum clock."""
    target_period = 30 * 60.0  # 30 minutes in seconds
    length = Pendulum.length_for_period(target_period)
    pend = Pendulum(length_m=length)
    clock = TrashClock(pend)

    print(f"Target period: {target_period:.1f} s")
    print(f"Computed pendulum length: {length:.2f} m")
    print(f"Pendulum period (computed): {pend.period():.2f} s")
    print(f"Clock tick interval (half period): {clock.tick_interval:.2f} s")

    # Simulate the first 5 ticks
    current = 0.0
    for i in range(5):
        nxt = clock.next_tick(current)
        print(f"Tick {i + 1}: {nxt:.2f} s")
        current = nxt + 0.01  # advance slightly beyond the tick


if __name__ == "__main__":
    # Run demo
    _demo()

    # Simple unit‑test style assertions
    # 1. Period of 1‑m pendulum ≈ 2.006 s
    p1 = Pendulum(length_m=1.0)
    assert abs(p1.period() - 2.006) < 0.005, "Period calculation error for 1 m pendulum"

    # 2. Length for 2.006‑s period should be ≈ 1 m
    l = Pendulum.length_for_period(p1.period())
    assert abs(l - 1.0) < 0.001, "Length calculation error for given period"

    # 3. Angle at t=0 should equal initial angle
    theta0 = 0.05
    assert abs(p1.angle_at_time(0.0, theta0) - theta0) < 1e-9, "Angle at t=0 mismatch"

    # 4. Simulation length correctness
    sim = p1.simulate(duration_s=1.0, dt_s=0.2, initial_angle_rad=theta0)
    assert len(sim) == 6, "Simulation step count mismatch"
    times, angles = zip(*sim)
    assert times[0] == 0.0 and times[-1] == 1.0, "Simulation time bounds error"

    # 5. TrashClock tick interval consistency
    clock = TrashClock(p1)
    assert abs(clock.tick_interval - p1.period() / 2) < 1e-9, "Tick interval mismatch"

    # 6. Next tick calculation
    assert clock.next_tick(0.0) == clock.tick_interval, "First tick error"
    assert clock.next_tick(clock.tick_interval) == 2 * clock.tick_interval, "Subsequent tick error"

    print("All internal tests passed.")