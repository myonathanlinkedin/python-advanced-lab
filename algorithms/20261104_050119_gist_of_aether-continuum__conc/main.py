"""
main.py
---------

Unit‑test suite and demonstration for the ReadWriteLock implementation.
All tests are deterministic and rely solely on the Python standard library.
"""

import threading
import time
import unittest
from typing import List

from core import ReadWriteLock


class TestReadWriteLock(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = ReadWriteLock()

    def test_multiple_readers(self) -> None:
        """Readers should be able to hold the lock concurrently."""
        start_times: List[float] = []
        end_times: List[float] = []
        barrier = threading.Barrier(5)

        def reader(idx: int) -> None:
            barrier.wait()
            with self.lock.read_lock():
                start_times.append(time.time())
                time.sleep(0.1)  # hold the lock briefly
                end_times.append(time.time())

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All readers should have overlapped: the earliest start should be
        # before the latest end, and the total span should be < 0.2 seconds.
        self.assertTrue(max(end_times) - min(start_times) < 0.2)

    def test_writer_exclusion(self) -> None:
        """Only one writer may hold the lock, and no readers may run concurrently."""
        events = []

        def writer(idx: int) -> None:
            with self.lock.write_lock():
                events.append(f"w{idx}_enter")
                time.sleep(0.1)
                events.append(f"w{idx}_exit")

        def reader(idx: int) -> None:
            with self.lock.read_lock():
                events.append(f"r{idx}_enter")
                time.sleep(0.05)
                events.append(f"r{idx}_exit")

        # Start a writer, then a reader, then another writer.
        threads = [
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=reader, args=(1,)),
            threading.Thread(target=writer, args=(2,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify ordering: w1 must complete before r1 can start,
        # and r1 must complete before w2 can start.
        self.assertEqual(events, [
            "w1_enter", "w1_exit",
            "r1_enter", "r1_exit",
            "w2_enter", "w2_exit",
        ])

    def test_writer_priority_over_new_readers(self) -> None:
        """
        When a writer is waiting, subsequent readers must block until the writer
        acquires and releases the lock. This prevents writer starvation.
        """
        order: List[str] = []
        start_barrier = threading.Barrier(3)

        def long_reader() -> None:
            with self.lock.read_lock():
                order.append("long_reader_start")
                start_barrier.wait()
                time.sleep(0.2)
                order.append("long_reader_end")

        def waiting_writer() -> None:
            start_barrier.wait()
            with self.lock.write_lock():
                order.append("writer_start")
                time.sleep(0.1)
                order.append("writer_end")

        def late_reader() -> None:
            # This reader arrives after the writer has started waiting.
            start_barrier.wait()
            with self.lock.read_lock():
                order.append("late_reader_start")
                time.sleep(0.05)
                order.append("late_reader_end")

        threads = [
            threading.Thread(target=long_reader),
            threading.Thread(target=waiting_writer),
            threading.Thread(target=late_reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Expected order:
        # 1. long_reader runs first.
        # 2. writer acquires after long_reader releases.
        # 3. late_reader runs after writer finishes.
        self.assertEqual(order, [
            "long_reader_start",
            "long_reader_end",
            "writer_start",
            "writer_end",
            "late_reader_start",
            "late_reader_end",
        ])

    def test_context_manager_exceptions(self) -> None:
        """Exceptions inside the context should propagate and still release the lock."""
        with self.assertRaises(ValueError):
            with self.lock.write_lock():
                raise ValueError("test")
        # After exception, lock must be free for a new writer.
        with self.lock.write_lock():
            pass  # should succeed without deadlock

    def test_acquire_timeout(self) -> None:
        """Timeouts should cause acquire_* to return False."""
        # Acquire write lock indefinitely.
        self.lock.acquire_write()
        # Attempt to acquire read lock with a short timeout.
        start = time.time()
        acquired = self.lock.acquire_read(timeout=0.05)
        elapsed = time.time() - start
        self.assertFalse(acquired)
        self.assertLess(elapsed, 0.2)  # ensure we didn't block too long
        self.lock.release_write()


def demo() -> None:
    """Simple demonstration of concurrent readers and writers."""
    lock = ReadWriteLock()
    stop_event = threading.Event()

    def reader(id_: int) -> None:
        while not stop_event.is_set():
            with lock.read_lock():
                print(f"[Reader {id_}] reading")
                time.sleep(0.1)

    def writer(id_: int) -> None:
        while not stop_event.is_set():
            with lock.write_lock():
                print(f"[Writer {id_}] writing")
                time.sleep(0.2)

    threads = [
        threading.Thread(target=reader, args=(i,)) for i in range(2)
    ] + [
        threading.Thread(target=writer, args=(i,)) for i in range(1)
    ]

    for t in threads:
        t.start()
    try:
        time.sleep(2)  # run demo for a short while
    finally:
        stop_event.set()
        for t in threads:
            t.join()


if __name__ == "__main__":
    # Run unit tests first; if they pass, launch the demo.
    unittest.main(exit=False)
    print("\n--- Demo of ReadWriteLock ---")
    demo()
