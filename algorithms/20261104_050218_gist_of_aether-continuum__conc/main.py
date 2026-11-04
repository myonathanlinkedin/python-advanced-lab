import unittest
import threading
import time
from typing import List

from core import BoundedBuffer, ThreadPool, Future


class TestBoundedBuffer(unittest.TestCase):
    def test_put_get_basic(self) -> None:
        buf = BoundedBuffer[int](capacity=3)
        buf.put(1)
        buf.put(2)
        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.get(), 1)
        self.assertEqual(buf.get(), 2)
        self.assertEqual(len(buf), 0)

    def test_try_put_get(self) -> None:
        buf = BoundedBuffer[int](capacity=1)
        self.assertTrue(buf.try_put(10))
        self.assertFalse(buf.try_put(20))  # buffer full
        self.assertEqual(buf.try_get(), 10)
        self.assertIsNone(buf.try_get())   # buffer empty

    def test_concurrent_producers_consumers(self) -> None:
        capacity = 5
        items_per_producer = 1000
        producer_count = 4
        consumer_count = 4

        buf = BoundedBuffer[int](capacity=capacity)
        produced: List[int] = []
        consumed: List[int] = []

        prod_lock = threading.Lock()
        cons_lock = threading.Lock()

        def producer(start: int) -> None:
            for i in range(start, start + items_per_producer):
                buf.put(i)
                with prod_lock:
                    produced.append(i)

        def consumer() -> None:
            while True:
                try:
                    item = buf.get()
                except Exception:
                    break
                with cons_lock:
                    consumed.append(item)
                # Stop condition when all items have been consumed
                if len(consumed) >= items_per_producer * producer_count:
                    break

        prod_threads = [threading.Thread(target=producer, args=(i * items_per_producer,)) for i in range(producer_count)]
        cons_threads = [threading.Thread(target=consumer) for _ in range(consumer_count)]

        for t in prod_threads + cons_threads:
            t.start()
        for t in prod_threads:
            t.join()
        # After producers finish, allow consumers to drain remaining items
        while len(consumed) < items_per_producer * producer_count:
            time.sleep(0.01)
        for t in cons_threads:
            t.join(timeout=0.1)

        self.assertCountEqual(produced, consumed)
        self.assertEqual(len(buf), 0)

    def test_buffer_timeout(self) -> None:
        buf = BoundedBuffer[int](capacity=1)
        self.assertTrue(buf.try_put(1, timeout=0.1))
        start = time.time()
        self.assertFalse(buf.try_put(2, timeout=0.2))
        elapsed = time.time() - start
        self.assertTrue(0.15 <= elapsed <= 0.3)

        self.assertEqual(buf.try_get(timeout=0.1), 1)
        start = time.time()
        self.assertIsNone(buf.try_get(timeout=0.2))
        elapsed = time.time() - start
        self.assertTrue(0.15 <= elapsed <= 0.3)


class TestThreadPool(unittest.TestCase):
    def test_submit_and_result(self) -> None:
        pool = ThreadPool(max_workers=3)

        def add(a: int, b: int) -> int:
            return a + b

        futures: List[Future[int]] = [pool.submit(add, i, i * 2) for i in range(10)]
        results = [f.result() for f in futures]
        expected = [i + i * 2 for i in range(10)]
        self.assertListEqual(results, expected)
        pool.shutdown()

    def test_exception_propagation(self) -> None:
        pool = ThreadPool(max_workers=2)

        def faulty() -> None:
            raise ValueError("intentional error")

        future = pool.submit(faulty)
        with self.assertRaises(ValueError):
            future.result()
        pool.shutdown()

    def test_shutdown_behavior(self) -> None:
        pool = ThreadPool(max_workers=2)
        barrier = threading.Barrier(3)

        def wait_and_return(x: int) -> int:
            barrier.wait()
            return x

        f1 = pool.submit(wait_and_return, 1)
        f2 = pool.submit(wait_and_return, 2)
        # Ensure both tasks have started
        barrier.wait()
        self.assertEqual(f1.result(), 1)
        self.assertEqual(f2.result(), 2)
        pool.shutdown(wait=True)
        with self.assertRaises(RuntimeError):
            pool.submit(lambda: 0)


if __name__ == '__main__':
    unittest.main()
