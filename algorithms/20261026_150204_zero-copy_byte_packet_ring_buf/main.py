"""
main.py
---------

Unit‑test suite and simple demonstration for ``RingBufferDispatcher``.
"""

import unittest
import time
from typing import List

from core import RingBufferDispatcher


class TestRingBufferDispatcher(unittest.TestCase):
    def test_initial_state(self):
        buf = RingBufferDispatcher(4)
        self.assertTrue(buf.is_empty())
        self.assertFalse(buf.is_full())
        self.assertEqual(len(buf), 0)

    def test_push_pop_cycle(self):
        buf = RingBufferDispatcher(3)
        packets = [b"alpha", b"beta", b"gamma"]
        for pkt in packets:
            self.assertTrue(buf.push(pkt))
        self.assertTrue(buf.is_full())
        self.assertFalse(buf.push(b"delta"))  # overflow

        # Verify zero‑copy: same object identity
        for original, stored in zip(packets, buf):
            self.assertIs(original, stored)

        # Pop in FIFO order
        for expected in packets:
            pkt = buf.pop()
            self.assertIsNotNone(pkt)
            self.assertEqual(pkt, expected)
        self.assertTrue(buf.is_empty())
        self.assertIsNone(buf.pop())  # underflow

    def test_peek(self):
        buf = RingBufferDispatcher(2)
        self.assertIsNone(buf.peek())
        buf.push(b"first")
        self.assertEqual(buf.peek(), b"first")
        self.assertEqual(len(buf), 1)  # peek does not remove

    def test_dispatch(self):
        buf = RingBufferDispatcher(5)
        data = [b"a", b"b", b"c"]
        for d in data:
            buf.push(d)

        collected: List[bytes] = []

        def handler(pkt: bytes) -> None:
            collected.append(pkt)

        dispatched = buf.dispatch(handler)
        self.assertEqual(dispatched, len(data))
        self.assertEqual(collected, data)
        self.assertTrue(buf.is_empty())

    def test_iterator_consistency(self):
        buf = RingBufferDispatcher(4)
        for i in range(3):
            buf.push(bytes([i]))
        self.assertEqual(list(buf), [b"\x00", b"\x01", b"\x02"])
        # After a pop, iterator reflects new state
        buf.pop()
        self.assertEqual(list(buf), [b"\x01", b"\x02"])

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            RingBufferDispatcher(0)

    def test_invalid_packet_type(self):
        buf = RingBufferDispatcher(2)
        with self.assertRaises(TypeError):
            buf.push("not-bytes")  # type: ignore[arg-type]

    def test_performance_benchmark(self):
        """A lightweight benchmark to ensure deterministic behaviour."""
        N = 100_000
        buf = RingBufferDispatcher(1024)
        start = time.perf_counter()
        for i in range(N):
            # Wrap around automatically due to capacity
            buf.push(bytes([i % 256]))
            if buf.is_full():
                buf.pop()
        elapsed = time.perf_counter() - start
        # The operation count is O(N); we assert it completes quickly.
        self.assertLess(elapsed, 2.0, f"Benchmark too slow: {elapsed:.2f}s")

if __name__ == "__main__":
    # Run unit tests with verbose output.
    unittest.main(verbosity=2)
