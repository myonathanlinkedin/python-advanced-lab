"""
Unit‑test suite and simple benchmark for ``StratumCacheRingBuffer``.
The tests exercise normal operation, edge cases, cache semantics and basic
thread‑safety.  Running the module directly executes the tests and prints a
tiny throughput benchmark.
"""

import threading
import time
import unittest
from typing import List

from core import StratumCacheRingBuffer


class TestStratumCacheRingBuffer(unittest.TestCase):
    def test_basic_put_get(self) -> None:
        buf = StratumCacheRingBuffer[int](capacity=5)
        for i in range(5):
            buf.put(i)
        self.assertTrue(buf.is_full())
        for i in range(5):
            self.assertEqual(buf.get(), i)
        self.assertTrue(buf.is_empty())

    def test_overflow(self) -> None:
        buf = StratumCacheRingBuffer[int](capacity=3)
        for i in range(3):
            buf.put(i)
        with self.assertRaises(BufferError):
            buf.put(99)

    def test_underflow(self) -> None:
        buf = StratumCacheRingBuffer[int](capacity=3)
        with self.assertRaises(BufferError):
            buf.get()
        buf.put(1)
        self.assertEqual(buf.get(), 1)
        with self.assertRaises(BufferError):
            buf.get()

    def test_cache_behavior(self) -> None:
        buf = StratumCacheRingBuffer[int](capacity=4, cache_size=2)
        for i in range(4):
            buf.put(i)          # cache now holds [3,2] (most recent first)
        self.assertEqual(buf.cache_contents(), [3, 2])

        # get() should first drain the cache
        self.assertEqual(buf.get(), 3)
        self.assertEqual(buf.get(), 2)
        # cache empty → fetch from ring buffer (oldest remaining = 0)
        self.assertEqual(buf.get(), 0)
        self.assertEqual(buf.get(), 1)
        self.assertTrue(buf.is_empty())

    def test_peek(self) -> None:
        buf = StratumCacheRingBuffer[int](capacity=3, cache_size=1)
        buf.put(10)
        buf.put(20)
        self.assertEqual(buf.peek(), 20)   # cache hit
        self.assertEqual(buf.get(), 20)    # consumes cache
        self.assertEqual(buf.peek(), 10)   # now head of ring buffer
        self.assertEqual(buf.get(), 10)

    def test_thread_safety(self) -> None:
        buf = StratumCacheRingBuffer[int](capacity=1000, cache_size=10)
        produced: List[int] = []
        consumed: List[int] = []

        def producer() -> None:
            for i in range(500):
                buf.put(i)
                produced.append(i)

        def consumer() -> None:
            read = 0
            while read < 500:
                try:
                    val = buf.get()
                    consumed.append(val)
                    read += 1
                except BufferError:
                    # buffer empty – spin briefly
                    time.sleep(0.0001)

        t_prod = threading.Thread(target=producer)
        t_cons = threading.Thread(target=consumer)
        t_prod.start()
        t_cons.start()
        t_prod.join()
        t_cons.join()

        self.assertCountEqual(produced, consumed)
        self.assertTrue(buf.is_empty())

    def test_snapshot_and_repr(self) -> None:
        buf = StratumCacheRingBuffer[str](capacity=3, cache_size=1)
        buf.put("a")
        buf.put("b")
        self.assertEqual(buf.buffer_snapshot(), ["a", "b"])
        self.assertIn("capacity=3", repr(buf))


def benchmark_throughput(iterations: int = 200_000) -> None:
    """
    Simple single‑threaded benchmark: insert and retrieve ``iterations`` items.
    The cache is disabled to measure raw ring‑buffer speed.
    """
    buf = StratumCacheRingBuffer[int](capacity=iterations)
    start = time.perf_counter()
    for i in range(iterations):
        buf.put(i)
    for _ in range(iterations):
        buf.get()
    elapsed = time.perf_counter() - start
    ops_per_sec = (iterations * 2) / elapsed
    print(f"Benchmark: {iterations:,} puts + gets in {elapsed:.4f}s "
          f"≈ {ops_per_sec:,.0f} ops/s")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run a quick benchmark
    print("\nRunning simple throughput benchmark (cache disabled)...")
    benchmark_throughput(200_000)
