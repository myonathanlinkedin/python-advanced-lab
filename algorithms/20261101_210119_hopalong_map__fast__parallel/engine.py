from __future__ import annotations

import math
from typing import List, Tuple

from .types import HopalongParams, Point, Checkpoint, OrbitGenerator, Orbit

def _sign(value: float) -> int:
    """Return the sign of a float: 1, -1 or 0."""
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0

def hopalong_step(state: Point, params: HopalongParams) -> Point:
    """
    Compute the next state of the Hopalong map.

    x_{n+1} = y_n - sign(x_n) * sqrt(|b * x_n - c|)
    y_{n+1} = a - x_n
    """
    x, y = state.x, state.y
    term = params.b * x - params.c
    next_x = y - _sign(x) * math.sqrt(abs(term))
    next_y = params.a - x
    return Point(next_x, next_y)

def generate_orbit(
    initial: Point,
    params: HopalongParams,
    steps: int,
) -> OrbitGenerator:
    """
    Yield the orbit points starting from ``initial`` for ``steps`` iterations.
    The initial point is yielded first (step 0).
    """
    if steps < 0:
        raise ValueError("steps must be non‑negative")
    state = initial
    for _ in range(steps + 1):
        yield state
        state = hopalong_step(state, params)

def compute_checkpoints(
    initial: Point,
    params: HopalongParams,
    total_steps: int,
    interval: int,
) -> List[Checkpoint]:
    """
    First pass: compute checkpoints every ``interval`` steps.
    Returns a list of (step_index, state) tuples.
    The checkpoint at step 0 (the initial state) is always included.
    """
    if interval <= 0:
        raise ValueError("interval must be positive")
    if total_steps < 0:
        raise ValueError("total_steps must be non‑negative")

    checkpoints: List[Checkpoint] = [(0, initial)]
    state = initial
    for step in range(1, total_steps + 1):
        state = hopalong_step(state, params)
        if step % interval == 0:
            checkpoints.append((step, state))
    # Ensure the final state is a checkpoint if it wasn't captured exactly at an interval
    if checkpoints[-1][0] != total_steps:
        checkpoints.append((total_steps, state))
    return checkpoints

def segment_from_checkpoint(
    checkpoint: Checkpoint,
    params: HopalongParams,
    segment_length: int,
    include_start: bool = True,
) -> Orbit:
    """
    Generate a segment of the orbit starting from ``checkpoint``.
    ``segment_length`` is the number of steps to advance *after* the checkpoint.
    If ``include_start`` is True, the checkpoint state itself is the first element.
    """
    start_index, start_state = checkpoint
    if segment_length < 0:
        raise ValueError("segment_length must be non‑negative")
    segment: Orbit = []
    state = start_state
    steps = segment_length + (1 if include_start else 0)
    for i in range(steps):
        if i == 0 and include_start:
            segment.append(state)
        else:
            state = hopalong_step(state, params)
            segment.append(state)
    return segment

def recombine_segments(
    checkpoints: List[Checkpoint],
    params: HopalongParams,
    total_steps: int,
    interval: int,
) -> Orbit:
    """
    Second pass: using the checkpoints, recompute the full orbit.
    This function mimics a parallel decomposition where each segment could be
    processed independently.
    """
    if not checkpoints:
        raise ValueError("checkpoints list cannot be empty")
    # Sort checkpoints by step index to guarantee order
    checkpoints = sorted(checkpoints, key=lambda cp: cp[0])

    full_orbit: Orbit = []
    for i, cp in enumerate(checkpoints):
        start_step = cp[0]
        # Determine how many steps this segment should cover
        if i + 1 < len(checkpoints):
            next_step = checkpoints[i + 1][0]
            segment_len = next_step - start_step
        else:
            # Last segment goes to total_steps
            segment_len = total_steps - start_step
        # For all but the first segment we omit the duplicated start point
        include_start = (i == 0)
        segment = segment_from_checkpoint(cp, params, segment_len, include_start)
        # Avoid duplicating the boundary point when concatenating
        if i > 0:
            segment = segment[1:]  # drop the first point (already present)
        full_orbit.extend(segment)
    return full_orbit
