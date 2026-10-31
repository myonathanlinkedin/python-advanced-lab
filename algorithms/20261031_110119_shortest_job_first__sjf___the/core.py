from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import heapq

@dataclass(order=True)
class Job:
    """Represents a CPU job with arrival and burst times."""
    arrival_time: int
    burst_time: int
    job_id: int = field(compare=False)

    def __post_init__(self) -> None:
        if self.arrival_time < 0:
            raise ValueError(f"Arrival time cannot be negative: {self.arrival_time}")
        if self.burst_time < 0:
            raise ValueError(f"Burst time cannot be negative: {self.burst_time}")

def schedule_jobs(jobs: List[Job]) -> Tuple[List[int], Dict[str, float]]:
    """
    Non‑preemptive Shortest Job First (SJF) scheduler.

    Parameters
    ----------
    jobs : List[Job]
        List of jobs to schedule.

    Returns
    -------
    Tuple[List[int], Dict[str, float]]
        A tuple containing:
        - The execution order of job IDs.
        - A dictionary with average waiting time and average turnaround time.
    """
    if not jobs:
        return [], {"average_waiting_time": 0.0, "average_turnaround_time": 0.0}

    # Sort jobs by arrival time for initial processing
    jobs_sorted = sorted(jobs, key=lambda j: j.arrival_time)
    ready_queue: List[Tuple[int, int, Job]] = []  # (burst_time, job_id, job)
    current_time = 0
    index = 0  # index into jobs_sorted
    order: List[int] = []
    total_waiting = 0
    total_turnaround = 0

    while index < len(jobs_sorted) or ready_queue:
        # If no ready jobs, jump to next arrival
        if not ready_queue:
            current_time = max(current_time, jobs_sorted[index].arrival_time)

        # Enqueue all jobs that have arrived by current_time
        while index < len(jobs_sorted) and jobs_sorted[index].arrival_time <= current_time:
            job = jobs_sorted[index]
            heapq.heappush(ready_queue, (job.burst_time, job.job_id, job))
            index += 1

        # Pop job with smallest burst time
        burst, _, job = heapq.heappop(ready_queue)
        order.append(job.job_id)

        waiting_time = current_time - job.arrival_time
        turnaround_time = waiting_time + job.burst_time

        total_waiting += waiting_time
        total_turnaround += turnaround_time

        current_time += job.burst_time

    n = len(jobs)
    metrics = {
        "average_waiting_time": total_waiting / n,
        "average_turnaround_time": total_turnaround / n,
    }
    return order, metrics
