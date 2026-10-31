from __future__ import annotations
import unittest
from typing import List
from core import Job, schedule_jobs

class TestSJF(unittest.TestCase):
    def test_empty(self):
        order, metrics = schedule_jobs([])
        self.assertEqual(order, [])
        self.assertEqual(metrics["average_waiting_time"], 0.0)
        self.assertEqual(metrics["average_turnaround_time"], 0.0)

    def test_single_job(self):
        job = Job(job_id=1, arrival_time=0, burst_time=5)
        order, metrics = schedule_jobs([job])
        self.assertEqual(order, [1])
        self.assertAlmostEqual(metrics["average_waiting_time"], 0.0)
        self.assertAlmostEqual(metrics["average_turnaround_time"], 5.0)

    def test_multiple_jobs_order(self):
        jobs = [
            Job(job_id=1, arrival_time=0, burst_time=8),
            Job(job_id=2, arrival_time=1, burst_time=4),
            Job(job_id=3, arrival_time=2, burst_time=9),
            Job(job_id=4, arrival_time=3, burst_time=5),
        ]
        order, metrics = schedule_jobs(jobs)
        self.assertEqual(order, [1, 2, 4, 3])
        self.assertAlmostEqual(metrics["average_waiting_time"], 3.75)
        self.assertAlmostEqual(metrics["average_turnaround_time"], 11.25)

    def test_same_arrival_times(self):
        jobs = [
            Job(job_id=1, arrival_time=0, burst_time=3),
            Job(job_id=2, arrival_time=0, burst_time=1),
            Job(job_id=3, arrival_time=0, burst_time=2),
        ]
        order, metrics = schedule_jobs(jobs)
        self.assertEqual(order, [2, 3, 1])
        self.assertAlmostEqual(metrics["average_waiting_time"], 1.0)
        self.assertAlmostEqual(metrics["average_turnaround_time"], 4.0)

    def test_unsorted_input(self):
        jobs = [
            Job(job_id=3, arrival_time=5, burst_time=2),
            Job(job_id=1, arrival_time=0, burst_time=4),
            Job(job_id=2, arrival_time=2, burst_time=1),
        ]
        order, metrics = schedule_jobs(jobs)
        self.assertEqual(order, [1, 2, 3])
        self.assertAlmostEqual(metrics["average_waiting_time"], 1.0)
        self.assertAlmostEqual(metrics["average_turnaround_time"], 4.0)

    def test_negative_times(self):
        with self.assertRaises(ValueError):
            Job(job_id=1, arrival_time=-1, burst_time=5)
        with self.assertRaises(ValueError):
            Job(job_id=2, arrival_time=0, burst_time=-3)

    def test_zero_burst(self):
        jobs = [
            Job(job_id=1, arrival_time=0, burst_time=0),
            Job(job_id=2, arrival_time=1, burst_time=2),
        ]
        order, metrics = schedule_jobs(jobs)
        self.assertEqual(order, [1, 2])
        self.assertAlmostEqual(metrics["average_waiting_time"], 0.5)
        self.assertAlmostEqual(metrics["average_turnaround_time"], 1.5)

    def test_large_job_set(self):
        jobs = [Job(job_id=i, arrival_time=i%5, burst_time=(i%3)+1) for i in range(1, 101)]
        order, metrics = schedule_jobs(jobs)
        self.assertEqual(len(order), 100)
        self.assertTrue(0 <= metrics["average_waiting_time"] <= 10)
        self.assertTrue(0 <= metrics["average_turnaround_time"] <= 20)

if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demo
    demo_jobs = [
        Job(job_id=1, arrival_time=0, burst_time=6),
        Job(job_id=2, arrival_time=2, burst_time=8),
        Job(job_id=3, arrival_time=4, burst_time=7),
        Job(job_id=4, arrival_time=6, burst_time=3),
    ]
    order, metrics = schedule_jobs(demo_jobs)
    print("\nDemo Schedule:")
    print(f"Execution order: {order}")
    print(f"Average waiting time: {metrics['average_waiting_time']:.2f}")
    print(f"Average turnaround time: {metrics['average_turnaround_time']:.2f}")
