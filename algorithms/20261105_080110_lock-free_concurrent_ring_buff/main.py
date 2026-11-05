"""
Unit‑test suite and demonstration for the lock‑free SPSC ring buffer.

The tests cover:
- Basic single‑threaded behaviour.
- Edge cases (full, empty, wrap‑around).
- Multi‑threaded producer/consumer correctness.
- Simple performance benchmark.
"""

import threading
import time
import unittest
from typing import List

from core import RingBuffer


class TestRingBuffer(unittest.TestCase):
    def test_initial_state(self) -> None:
        rb = RingBuffer[int](capacity=8)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())
        self.assertEqual(len(rb), 0)
        self.assertIsNone(rb.pop())

    def test_push_pop_basic(self) -> None:
        rb = RingBuffer[int](capacity=4)  # usable slots = 3
        self.assertTrue(rb.push(1))
        self.assertTrue(rb.push(2))
        self.assertTrue(rb.push(3))
        self.assertFalse(rb.push(99))  # buffer full
        self.assertTrue(rb.is_full())
        self.assertEqual(len(rb), 3)

        self.assertEqual(rb.pop(), 1)
        self.assertEqual(rb.pop(), 2)
        self.assertEqual(rb.pop(), 3)
        self.assertTrue(rb.is_empty())
        self.assertIsNone(rb.pop())

    def test_wrap_around(self) -> None:
        rb = RingBuffer[int](capacity=8)  # usable slots = 7
        for i in range(7):
            self.assertTrue(rb.push(i))
        self.assertTrue(rb.is_full())
        # Pop three items, freeing space at the front.
        for i in range(3):
            self.assertEqual(rb.pop(), i)
        self.assertFalse(rb.is_full())
        # Push again to cause wrap‑around.
        for i in range(3):
            self.assertTrue(rb.push(100 + i))
        # Verify order of remaining items.
        expected = [3, 4, 5, 6, 100, 101, 102]
        for exp in expected:
            self.assertEqual(rb.pop(), exp)
        self.assertTrue(rb.is_empty())

    def test_multi_threaded_spsc(self) -> None:
        capacity = 1024
        rb = RingBuffer[int](capacity=capacity)

        n_items = 100_000
        produced: List[int] = []
        consumed: List[int] = []

        def producer() -> None:
            for i in range(n_items):
                # Busy‑wait until there is space.
                while not rb.push(i):
                    continue
                produced.append(i)

        def consumer() -> None:
            count = 0
            while count < n_items:
                item = rb.pop()
                if item is None:
                    continue
                consumed.append(item)
                count += 1

        t_prod = threading.Thread(target=producer, name="producer")
        t_cons = threading.Thread(target=consumer, name="consumer")
        t_prod.start()
        t_cons.start()
        t_prod.join()
        t_cons.join()

        self.assertEqual(len(produced), n_items)
        self.assertEqual(len(consumed), n_items)
        self.assertEqual(produced, consumed)  # FIFO order preserved

    def test_performance_benchmark(self) -> None:
        """A very lightweight benchmark to ensure O(1) behaviour."""
        rb = RingBuffer[int](capacity=2 ** 12)  # 4096 slots
        n = 1_000_000
        start = time.perf_counter()
        for i in range(n):
            while not rb.push(i):
                pass
        for _ in range(n):
            while rb.pop() is None:
                pass
        elapsed = time.perf_counter() - start
        # The benchmark is not strict; we only assert it finishes quickly.
        self.assertLess(elapsed, 5.0, f"Benchmark too slow: {elapsed:.2f}s")


if __name__ == "__main__":
    # Run the unit tests with a higher verbosity for demonstration.
    unittest.main(verbosity=2)
