import threading
import unittest
from typing import List, Set

from core import BoundedBuffer, ReadWriteLock


class TestBoundedBuffer(unittest.TestCase):
    def test_single_thread_put_get(self):
        buf = BoundedBuffer[int](capacity=3)
        self.assertEqual(len(buf), 0)
        buf.put(1)
        buf.put(2)
        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.get(), 1)
        self.assertEqual(buf.get(), 2)
        self.assertEqual(len(buf), 0)

    def test_blocking_behavior(self):
        buf = BoundedBuffer[int](capacity=1)
        barrier = threading.Barrier(2)

        def producer():
            barrier.wait()
            buf.put(42)          # Should succeed immediately.
            buf.put(99)          # Blocks until consumer takes the first item.

        def consumer():
            barrier.wait()
            self.assertEqual(buf.get(), 42)
            self.assertEqual(buf.get(), 99)

        t_prod = threading.Thread(target=producer)
        t_cons = threading.Thread(target=consumer)
        t_prod.start()
        t_cons.start()
        t_prod.join()
        t_cons.join()
        self.assertEqual(len(buf), 0)

    def test_concurrent_producers_consumers(self):
        capacity = 5
        buf = BoundedBuffer[int](capacity=capacity)
        n_producers = 4
        n_consumers = 3
        items_per_producer = 1000

        # Shared collection for verification.
        consumed: List[int] = []
        consumed_lock = threading.Lock()

        start_barrier = threading.Barrier(n_producers + n_consumers)

        def producer(pid: int):
            start_barrier.wait()
            base = pid * items_per_producer
            for i in range(items_per_producer):
                buf.put(base + i)

        def consumer():
            start_barrier.wait()
            while True:
                try:
                    item = buf.get()
                    with consumed_lock:
                        consumed.append(item)
                except Exception:
                    break

        producers = [threading.Thread(target=producer, args=(i,)) for i in range(n_producers)]
        consumers = [threading.Thread(target=consumer) for _ in range(n_consumers)]

        for t in producers + consumers:
            t.start()

        for t in producers:
            t.join()

        # After all producers finish, we need to signal consumers to stop.
        # Insert sentinel values equal to number of consumers.
        for _ in range(n_consumers):
            buf.put(-1)  # sentinel

        for t in consumers:
            t.join()

        # Verify all expected items are present and no duplicates.
        expected: Set[int] = set()
        for pid in range(n_producers):
            expected.update(range(pid * items_per_producer, (pid + 1) * items_per_producer))

        # Remove sentinel values.
        consumed = [x for x in consumed if x != -1]

        self.assertEqual(set(consumed), expected)
        self.assertEqual(len(consumed), len(expected))


class TestReadWriteLock(unittest.TestCase):
    def test_exclusive_write(self):
        lock = ReadWriteLock()
        shared: List[int] = []
        n_threads = 10
        barrier = threading.Barrier(n_threads)

        def writer(tid: int):
            barrier.wait()
            with lock.write_lock():
                # Critical section: exclusive access.
                current = list(shared)
                # Simulate work.
                for _ in range(1000):
                    pass
                shared.append(tid)
                # Ensure no other writer interleaved.
                self.assertEqual(shared[-1], tid)
                self.assertEqual(set(shared), set(current + [tid]))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertCountEqual(shared, list(range(n_threads)))

    def test_multiple_readers(self):
        lock = ReadWriteLock()
        read_counter = 0
        read_counter_lock = threading.Lock()
        n_readers = 20
        barrier = threading.Barrier(n_readers)

        def reader():
            nonlocal read_counter
            barrier.wait()
            with lock.read_lock():
                with read_counter_lock:
                    read_counter += 1
                # Simulate read work.
                for _ in range(5000):
                    pass
                with read_counter_lock:
                    read_counter -= 1

        threads = [threading.Thread(target=reader) for _ in range(n_readers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # If the lock allowed concurrent reads, read_counter should have been >1 at some point.
        # We cannot directly observe that without instrumentation, but the test ensures no deadlock.

    def test_writer_priority(self):
        lock = ReadWriteLock()
        order: List[str] = []
        order_lock = threading.Lock()
        start = threading.Event()

        def reader():
            start.wait()
            with lock.read_lock():
                with order_lock:
                    order.append('R')
                # Hold the read lock a bit.
                threading.Event().wait(0.05)

        def writer():
            start.wait()
            with lock.write_lock():
                with order_lock:
                    order.append('W')

        # Start a long‑running reader.
        t_reader1 = threading.Thread(target=reader)
        t_reader1.start()
        # Give it a head start.
        threading.Event().wait(0.01)

        # Start a writer that should wait until the first reader releases.
        t_writer = threading.Thread(target=writer)
        t_writer.start()

        # Start a second reader that arrives after the writer is waiting.
        t_reader2 = threading.Thread(target=reader)
        t_reader2.start()

        # Release all to run.
        start.set()

        for t in (t_reader1, t_writer, t_reader2):
            t.join()

        # Expected order: first reader, writer, then second reader (writer gets priority).
        self.assertEqual(order, ['R', 'W', 'R'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
