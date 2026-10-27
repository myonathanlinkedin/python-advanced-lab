"""
Unit‑test suite and demonstration for `BoundedBlockingQueue`.
All tests are deterministic and avoid flaky timing by using explicit
synchronisation primitives.
"""

import threading
import time
import unittest
from typing import List

from core import BoundedBlockingQueue


class TestBoundedBlockingQueue(unittest.TestCase):
    def test_basic_put_get(self) -> None:
        q = BoundedBlockingQueue[int](maxsize=3)
        q.put(1)
        q.put(2)
        self.assertEqual(q.size(), 2)
        self.assertEqual(q.get(), 1)
        self.assertEqual(q.get(), 2)
        self.assertEqual(q.size(), 0)

    def test_put_blocks_when_full(self) -> None:
        q = BoundedBlockingQueue[int](maxsize=2)
        q.put(1)
        q.put(2)

        # Event signals that the producer attempted to put a third item.
        blocked = threading.Event()
        finished = threading.Event()

        def producer():
            blocked.set()
            q.put(3)  # Should block until a consumer removes an item.
            finished.set()

        t = threading.Thread(target=producer)
        t.start()

        # Ensure the producer is blocked on the full queue.
        blocked.wait(timeout=1.0)
        self.assertFalse(finished.is_set(), "Producer should be blocked while queue is full")

        # Consume one item to free space.
        self.assertEqual(q.get(), 1)

        # Now the producer must finish promptly.
        finished.wait(timeout=1.0)
        self.assertTrue(finished.is_set(), "Producer should have completed after space became available")
        t.join()

        # Verify queue state.
        self.assertEqual(q.size(), 2)
        self.assertEqual(q.get(), 2)
        self.assertEqual(q.get(), 3)

    def test_get_blocks_when_empty(self) -> None:
        q = BoundedBlockingQueue[int](maxsize=2)

        started = threading.Event()
        finished = threading.Event()

        def consumer():
            started.set()
            val = q.get()  # Should block until an item is put.
            self.assertEqual(val, 42)
            finished.set()

        t = threading.Thread(target=consumer)
        t.start()

        started.wait(timeout=1.0)
        self.assertFalse(finished.is_set(), "Consumer should be blocked on empty queue")

        # Produce an item after a short pause.
        time.sleep(0.1)
        q.put(42)

        finished.wait(timeout=1.0)
        self.assertTrue(finished.is_set(), "Consumer should have received the item")
        t.join()

    def test_multiple_producers_consumers(self) -> None:
        capacity = 5
        q = BoundedBlockingQueue[int](maxsize=capacity)

        produced: List[int] = []
        consumed: List[int] = []
        produce_lock = threading.Lock()
        consume_lock = threading.Lock()
        n_items = 100

        def producer(start: int, step: int) -> None:
            for i in range(start, n_items, step):
                q.put(i)
                with produce_lock:
                    produced.append(i)

        def consumer() -> None:
            while True:
                try:
                    item = q.get(timeout=0.2)
                except TimeoutError:
                    # No more items expected after producers finish.
                    break
                with consume_lock:
                    consumed.append(item)

        prod_threads = [threading.Thread(target=producer, args=(i, 2)) for i in range(2)]
        cons_threads = [threading.Thread(target=consumer) for _ in range(3)]

        for t in prod_threads + cons_threads:
            t.start()
        for t in prod_threads:
            t.join()
        # After producers are done, consumers will eventually timeout and exit.
        for t in cons_threads:
            t.join()

        self.assertCountEqual(produced, consumed,
                              "All produced items must be consumed exactly once")
        self.assertLessEqual(q.size(), capacity,
                             "Queue size must never exceed its capacity")

    def test_timeout_put_and_get(self) -> None:
        q = BoundedBlockingQueue[int](maxsize=1)
        q.put(1)

        with self.assertRaises(TimeoutError):
            q.put(2, timeout=0.05)

        q.get()
        with self.assertRaises(TimeoutError):
            q.get(timeout=0.05)


def demo() -> None:
    """Simple interactive demonstration of the queue."""
    q = BoundedBlockingQueue[int](maxsize=3)

    def producer():
        for i in range(5):
            print(f"Producer putting {i}")
            q.put(i)
            time.sleep(0.2)

    def consumer():
        for _ in range(5):
            val = q.get()
            print(f"Consumer got {val}")
            time.sleep(0.35)

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


if __name__ == '__main__':
    # Run unit tests.
    unittest.main(exit=False)

    # Run a brief demo.
    print("\n--- Demo of BoundedBlockingQueue ---")
    demo()
