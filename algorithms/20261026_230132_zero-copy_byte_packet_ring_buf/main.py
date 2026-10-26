import time
import unittest
from typing import List

from core import Dispatcher, RingBuffer, Callback


class TestRingBuffer(unittest.TestCase):
    def test_basic_push_pop(self) -> None:
        rb = RingBuffer(3)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())

        data1 = b"abc"
        data2 = bytearray(b"def")
        data3 = memoryview(b"ghi")

        self.assertTrue(rb.push(data1))
        self.assertTrue(rb.push(data2))
        self.assertFalse(rb.is_empty())
        self.assertFalse(rb.is_full())

        # Zero‑copy verification: the returned view should reference the original.
        pkt = rb.pop()
        self.assertIsNotNone(pkt)
        self.assertEqual(pkt.tobytes(), data1)
        self.assertIs(pkt.obj, data1)  # type: ignore[attr-defined]

        self.assertTrue(rb.push(data3))
        self.assertTrue(rb.is_full())
        self.assertFalse(rb.push(b"overflow"))  # buffer full

        # Pop remaining items in order.
        self.assertEqual(rb.pop().tobytes(), data2)
        self.assertEqual(rb.pop().tobytes(), data3.tobytes())
        self.assertTrue(rb.is_empty())
        self.assertIsNone(rb.pop())

    def test_snapshot(self) -> None:
        rb = RingBuffer(5)
        packets = [b"a", b"bb", b"ccc"]
        for p in packets:
            rb.push(p)
        snap = rb.snapshot()
        self.assertEqual([mv.tobytes() for mv in snap], packets)

    def test_thread_safety(self) -> None:
        # Simple sanity check: concurrent pushes/pops should not corrupt state.
        rb = RingBuffer(1000)
        def producer():
            for i in range(500):
                rb.push(bytes([i % 256]))
        def consumer(collected: List[int]):
            while len(collected) < 500:
                pkt = rb.pop()
                if pkt is not None:
                    collected.append(pkt[0])

        prod_thread = threading.Thread(target=producer)
        cons_result: List[int] = []
        cons_thread = threading.Thread(target=consumer, args=(cons_result,))
        prod_thread.start()
        cons_thread.start()
        prod_thread.join()
        cons_thread.join()
        self.assertEqual(len(cons_result), 500)

class TestDispatcher(unittest.TestCase):
    def test_dispatch(self) -> None:
        rb = RingBuffer(4)
        disp = Dispatcher(rb)

        received: List[bytes] = []

        def handler_a(pkt: memoryview) -> None:
            received.append(pkt.tobytes())

        def handler_b(pkt: memoryview) -> None:
            received.append(pkt.tobytes())

        disp.register(0x01, handler_a)
        disp.register(0x02, handler_b)

        # Packets: first byte = type id
        rb.push(b"\x01hello")
        rb.push(b"\x02world")
        rb.push(b"\x01again")

        self.assertEqual(disp.dispatch_all(), 3)
        self.assertEqual(received, [b"\x01hello", b"\x02world", b"\x01again"])

    def test_missing_handler(self) -> None:
        rb = RingBuffer(2)
        disp = Dispatcher(rb)
        rb.push(b"\x03nop")
        with self.assertRaises(KeyError):
            disp.dispatch_one()

    def test_empty_packet(self) -> None:
        rb = RingBuffer(1)
        disp = Dispatcher(rb)
        disp.register(0x00, lambda _: None)
        rb.push(b"")  # zero‑length packet
        with self.assertRaises(ValueError):
            disp.dispatch_one()

def benchmark_dispatch(num_packets: int = 100_000) -> float:
    """Simple micro‑benchmark: push then dispatch ``num_packets``."""
    rb = RingBuffer(1024)
    disp = Dispatcher(rb)

    # Register a trivial handler that does nothing.
    disp.register(0xAA, lambda _: None)

    # Fill the ring buffer (wrapping around as needed)
    for _ in range(num_packets):
        # packet consists of type byte + 3 payload bytes
        rb.push(b"\xAAxyz")
        if rb.is_full():
            # Dispatch a batch to make room.
            disp.dispatch_all()

    # Drain any remaining packets.
    start = time.perf_counter()
    disp.dispatch_all()
    elapsed = time.perf_counter() - start
    return elapsed

class Demo(unittest.TestCase):
    def test_demo_run(self) -> None:
        """Run a tiny demo to ensure the module works end‑to‑end."""
        elapsed = benchmark_dispatch(10_000)
        # The benchmark should complete quickly; we only assert it ran.
        self.assertIsInstance(elapsed, float)

if __name__ == "__main__":
    unittest.main()
