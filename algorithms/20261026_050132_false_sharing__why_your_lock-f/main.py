"""
main.py
---------

Unit‑test suite and simple benchmark for the counters defined in core.py.
The tests verify functional correctness under concurrent access and
illustrate the performance impact of false sharing.
"""

import time
import unittest
from typing import Tuple

from core import MutexCounter, ShardedCounter, run_counter


class TestCounters(unittest.TestCase):
    def _run_and_check(self, counter_factory, threads: int, increments: int) -> None:
        """Helper that runs a counter and asserts the final value."""
        counter = counter_factory()
        run_counter(counter, increments_per_thread=increments, thread_count=threads)
        expected = threads * increments
        self.assertEqual(counter.value(), expected,
                         f"Counter total {counter.value()} != expected {expected}")

    def test_mutex_counter_correctness(self):
        """MutexCounter must produce the exact total under concurrency."""
        self._run_and_check(lambda: MutexCounter(), threads=8, increments=10_000)

    def test_sharded_counter_unpadded_correctness(self):
        """Unpadded ShardedCounter must be correct."""
        self._run_and_check(lambda: ShardedCounter(num_shards=8, padded=False),
                            threads=8, increments=10_000)

    def test_sharded_counter_padded_correctness(self):
        """Padded ShardedCounter must be correct."""
        self._run_and_check(lambda: ShardedCounter(num_shards=8, padded=True),
                            threads=8, increments=10_000)

    def test_false_sharing_impact(self):
        """
        Rough benchmark: unpadded sharded counter vs padded sharded counter.
        The padded version should not be slower; on many machines it is faster.
        """
        threads = 16
        increments = 200_000

        def bench(counter) -> float:
            start = time.perf_counter()
            run_counter(counter, increments_per_thread=increments, thread_count=threads)
            return time.perf_counter() - start

        unpadded = ShardedCounter(num_shards=threads, padded=False)
        padded = ShardedCounter(num_shards=threads, padded=True)

        t_unpadded = bench(unpadded)
        t_padded = bench(padded)

        # The benchmark is nondeterministic; we only assert that the padded
        # version does not take dramatically longer (e.g., >2× slower).
        self.assertLessEqual(t_padded, t_unpadded * 2,
                             f"Padded version too slow: {t_padded:.4f}s vs {t_unpadded:.4f}s")

        # Print timings for manual inspection (allowed in tests).
        print(f"\nBenchmark (threads={threads}, increments/thread={increments}):")
        print(f"  Unpadded sharded counter: {t_unpadded:.4f} s")
        print(f"  Padded   sharded counter: {t_padded:.4f} s")


if __name__ == '__main__':
    # Run the unit‑test suite.  The benchmark output appears on stdout.
    unittest.main(verbosity=2)
