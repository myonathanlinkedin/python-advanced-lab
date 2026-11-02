import unittest
import threading
import time
import queue
from core import BoundedBlockingQueue

class TestBoundedBlockingQueue(unittest.TestCase):
    def test_basic_put_get(self):
        q = BoundedBlockingQueue[int](3)
        q.put(1)
        q.put(2)
        self.assertEqual(q.qsize(), 2)
        self.assertEqual(q.get(), 1)
        self.assertEqual(q.get(), 2)
        self.assertTrue(q.empty())

    def test_full_behavior(self):
        q = BoundedBlockingQueue[int](2)
        q.put(10)
        q.put(20)
        self.assertTrue(q.full())
        with self.assertRaises(queue.Full):
            q.put(30, block=False)

    def test_empty_behavior(self):
        q = BoundedBlockingQueue[int](2)
        with self.assertRaises(queue.Empty):
            q.get(block=False)

    def test_blocking_put_timeout(self):
        q = BoundedBlockingQueue[int](1)
        q.put('a')
        start = time.monotonic()
        with self.assertRaises(queue.Full):
            q.put('b', timeout=0.2)
        elapsed = time.monotonic() - start
        self.assertTrue(0.15 <= elapsed <= 0.3)

    def test_blocking_get_timeout(self):
        q = BoundedBlockingQueue[int](1)
        start = time.monotonic()
        with self.assertRaises(queue.Empty):
            q.get(timeout=0.2)
        elapsed = time.monotonic() - start
        self.assertTrue(0.15 <= elapsed <= 0.3)

    def test_multi_threaded_producer_consumer(self):
        capacity = 5
        items = list(range(20))
        q = BoundedBlockingQueue[int](capacity)
        produced = []
        consumed = []

        def producer():
            for item in items:
                q.put(item)
                produced.append(item)

        def consumer():
            for _ in items:
                val = q.get()
                consumed.append(val)

        t_prod = threading.Thread(target=producer)
        t_cons = threading.Thread(target=consumer)
        t_prod.start()
        t_cons.start()
        t_prod.join()
        t_cons.join()

        self.assertCountEqual(produced, items)
        self.assertCountEqual(consumed, items)
        self.assertTrue(q.empty())

    def test_repr_and_len(self):
        q = BoundedBlockingQueue[int](10)
        self.assertEqual(repr(q), "BoundedBlockingQueue(maxsize=10, size=0)")
        q.put(1)
        self.assertEqual(len(q), 1)
        self.assertIn("size=1", repr(q))

if __name__ == '__main__':
    unittest.main(verbosity=2)
