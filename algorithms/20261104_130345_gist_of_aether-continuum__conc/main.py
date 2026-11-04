"""
main.py
-------
Unit tests, benchmarks, and entrypoint for the RingBuffer concurrency module.
"""

from __future__ import annotations

import threading
import time
import unittest

from core import Consumer, Producer, RingBuffer


class TestRingBufferBasics(unittest.TestCase):
    """Single-threaded correctness tests."""

    def test_capacity_validation(self) -> None:
        with self.assertRaises(ValueError):
            RingBuffer(0)
        with self.assertRaises(ValueError):
            RingBuffer(-1)

    def test_initial_state(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        self.assertEqual(len(buf), 0)
        self.assertTrue(buf.is_empty)
        self.assertFalse(buf.is_full)
        self.assertEqual(buf.capacity, 4)
        self.assertIsNone(buf.peek())
        self.assertIsNone(buf.dequeue(timeout=0.01))

    def test_enqueue_dequeue_fifo(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        for i in range(4):
            self.assertTrue(buf.enqueue(i))
        self.assertTrue(buf.is_full)
        self.assertEqual(len(buf), 4)
        self.assertEqual(buf.peek(), 0)
        for i in range(4):
            self.assertEqual(buf.dequeue(timeout=0.01), i)
        self.assertTrue(buf.is_empty)

    def test_wraparound(self) -> None:
        buf: RingBuffer[int] = RingBuffer(3)
        # Fill and drain multiple times to exercise wrap-around
        for cycle in range(5):
            for i in range(3):
                buf.enqueue(cycle * 10 + i)
            for i in range(3):
                self.assertEqual(buf.dequeue(timeout=0.01), cycle * 10 + i)
        self.assertTrue(buf.is_empty)

    def test_drain(self) -> None:
        buf: RingBuffer[int] = RingBuffer(5)
        for i in range(5):
            buf.enqueue(i)
        partial = buf.drain(3)
        self.assertEqual(partial, [0, 1, 2])
        self.assertEqual(len(buf), 2)
        rest = buf.drain()
        self.assertEqual(rest, [3, 4])
        self.assertTrue(buf.is_empty)

    def test_enqueue_timeout_when_full(self) -> None:
        buf: RingBuffer[int] = RingBuffer(2)
        buf.enqueue(1)
        buf.enqueue(2)
        start = time.monotonic()
        result = buf.enqueue(3, timeout=0.05)
        elapsed = time.monotonic() - start
        self.assertFalse(result)
        self.assertGreaterEqual(elapsed, 0.04)

    def test_dequeue_timeout_when_empty(self) -> None:
        buf: RingBuffer[int] = RingBuffer(2)
        start = time.monotonic()
        result = buf.dequeue(timeout=0.05)
        elapsed = time.monotonic() - start
        self.assertIsNone(result)
        self.assertGreaterEqual(elapsed, 0.04)


class TestRingBufferConcurrency(unittest.TestCase):
    """Multi-threaded stress tests."""

    def test_single_producer_single_consumer(self) -> None:
        N = 10_000
        buf: RingBuffer[int] = RingBuffer(64)
        received: list[int] = []
        lock = threading.Lock()

        def producer() -> None:
            for i in range(N):
                buf.enqueue(i)

        def consumer() -> None:
            for _ in range(N):
                item = buf.dequeue()
                assert item is not None
                with lock:
                    received.append(item)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual(received, list(range(N)))

    def test_multi_producer_single_consumer(self) -> None:
        N = 5_000
        NUM_PRODUCERS = 4
        buf: RingBuffer[int] = RingBuffer(128)
        received: list[int] = []
        lock = threading.Lock()
        total = N * NUM_PRODUCERS

        def producer(pid: int) -> None:
            for i in range(N):
                buf.enqueue(pid * N + i)

        def consumer() -> None:
            for _ in range(total):
                item = buf.dequeue()
                assert item is not None
                with lock:
                    received.append(item)

        threads = [threading.Thread(target=producer, args=(p,)) for p in range(NUM_PRODUCERS)]
        ct = threading.Thread(target=consumer)
        ct.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ct.join()
        self.assertEqual(len(received), total)
        self.assertEqual(sorted(received), list(range(total)))

    def test_single_producer_multi_consumer(self) -> None:
        N = 5_000
        NUM_CONSUMERS = 4
        buf: RingBuffer[int] = RingBuffer(128)
        received: list[int] = []
        lock = threading.Lock()

        def producer() -> None:
            for i in range(N):
                buf.enqueue(i)

        def consumer() -> None:
            while True:
                item = buf.dequeue(timeout=0.5)
                if item is None:
                    break
                with lock:
                    received.append(item)

        pt = threading.Thread(target=producer)
        ct = [threading.Thread(target=consumer) for _ in range(NUM_CONSUMERS)]
        pt.start()
        for t in ct:
            t.start()
        pt.join()
        for t in ct:
            t.join()
        self.assertEqual(len(received), N)
        self.assertEqual(sorted(received), list(range(N)))

    def test_producer_consumer_helpers(self) -> None:
        N = 1_000
        buf: RingBuffer[int] = RingBuffer(32)
        prod = Producer(buf, range(N))
        cons = Consumer(buf)
        prod.start()
        cons.start()
        time.sleep(0.5)
        prod.stop()
        cons.stop()
        self.assertEqual(len(cons.collected), N)
        self.assertEqual(sorted(cons.collected), list(range(N)))


class TestRingBufferBenchmark(unittest.TestCase):
    """Lightweight throughput benchmark (informational, not a pass/fail gate)."""

    def test_throughput(self) -> None:
        N = 50_000
        buf: RingBuffer[int] = RingBuffer(256)

        def producer() -> None:
            for i in range(N):
                buf.enqueue(i)

        def consumer() -> None:
            for _ in range(N):
                buf.dequeue()

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        start = time.monotonic()
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        elapsed = time.monotonic() - start
        ops_per_sec = N / elapsed if elapsed > 0 else float("inf")
        print(f"\n[Benchmark] {N} ops in {elapsed:.4f}s -> {ops_per_sec:,.0f} ops/s")


def main() -> None:
    """Run all tests and print a summary."""
    print("=" * 60)
    print("  RingBuffer Concurrency – Unit Test Suite")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__("main"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
