import unittest
import time
import threading
from core import RingBuffer

class TestRingBuffer(unittest.TestCase):
    def test_basic_operations(self):
        rb = RingBuffer[int](3)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())
        rb.push(1)
        rb.push(2)
        self.assertEqual(rb.size(), 2)
        self.assertEqual(rb.peek(), 1)
        self.assertEqual(rb.pop(), 1)
        self.assertEqual(rb.pop(), 2)
        self.assertTrue(rb.is_empty())

    def test_overflow(self):
        rb = RingBuffer[int](2)
        rb.push(10)
        rb.push(20)
        with self.assertRaises(OverflowError):
            rb.push(30)

    def test_underflow(self):
        rb = RingBuffer[int](2)
        with self.assertRaises(IndexError):
            rb.pop()
        rb.push(5)
        rb.pop()
        with self.assertRaises(IndexError):
            rb.pop()

    def test_iteration(self):
        rb = RingBuffer[str](4)
        for ch in "abcd":
            rb.push(ch)
        self.assertEqual(list(rb), ['a', 'b', 'c', 'd'])
        rb.pop()
        self.assertEqual(list(rb), ['b', 'c', 'd'])

    def test_thread_safety(self):
        rb = RingBuffer[int](1000)
        def producer():
            for i in range(5000):
                try:
                    rb.push(i)
                except OverflowError:
                    pass

        def consumer():
            count = 0
            while count < 5000:
                try:
                    rb.pop()
                    count += 1
                except IndexError:
                    pass

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertTrue(rb.is_empty() or rb.size() <= rb.capacity())

class Benchmark:
    def __init__(self, capacity: int, ops: int):
        self.capacity = capacity
        self.ops = ops

    def run(self):
        rb = RingBuffer[int](self.capacity)
        start = time.perf_counter()
        for i in range(self.ops):
            try:
                rb.push(i)
            except OverflowError:
                rb.pop()
                rb.push(i)
        end = time.perf_counter()
        duration = end - start
        print(f"Performed {self.ops} push/pop ops in {duration:.4f}s "
              f"({self.ops/duration:.2f} ops/s)")

if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demo benchmark
    bench = Benchmark(capacity=1024, ops=100000)
    bench.run()
