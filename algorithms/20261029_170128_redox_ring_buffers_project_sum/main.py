import unittest
import timeit
from core import RingBuffer

class TestRingBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.capacity = 5
        self.buffer = RingBuffer[int](self.capacity)

    def test_initial_state(self):
        self.assertTrue(self.buffer.is_empty())
        self.assertFalse(self.buffer.is_full())
        self.assertEqual(len(self.buffer), 0)

    def test_push_and_pop(self):
        for i in range(self.capacity):
            self.buffer.push(i)
        self.assertTrue(self.buffer.is_full())
        for i in range(self.capacity):
            self.assertEqual(self.buffer.pop(), i)
        self.assertTrue(self.buffer.is_empty())

    def test_overflow(self):
        for i in range(self.capacity):
            self.buffer.push(i)
        with self.assertRaises(OverflowError):
            self.buffer.push(99)

    def test_underflow(self):
        with self.assertRaises(IndexError):
            self.buffer.pop()
        with self.assertRaises(IndexError):
            self.buffer.peek()

    def test_peek(self):
        self.buffer.push(10)
        self.assertEqual(self.buffer.peek(), 10)
        self.assertEqual(len(self.buffer), 1)

    def test_iteration(self):
        items = [1, 2, 3]
        self.buffer.extend(items)
        self.assertEqual(list(self.buffer), items)

    def test_clear(self):
        self.buffer.extend([1, 2, 3])
        self.buffer.clear()
        self.assertTrue(self.buffer.is_empty())
        self.assertEqual(len(self.buffer), 0)

    def test_size_and_capacity(self):
        self.assertEqual(self.buffer.size(), 0)
        self.buffer.push(5)
        self.assertEqual(self.buffer.size(), 1)
        self.assertEqual(self.buffer.capacity, self.capacity)

    def test_getitem(self):
        self.buffer.extend([10, 20, 30])
        self.assertEqual(self.buffer[0], 10)
        self.assertEqual(self.buffer[1], 20)
        self.assertEqual(self.buffer[2], 30)
        self.assertEqual(self.buffer[-1], 30)
        self.assertEqual(self.buffer[-2], 20)
        with self.assertRaises(IndexError):
            _ = self.buffer[3]

    def test_capacity_one(self):
        buf = RingBuffer[int](1)
        buf.push(42)
        self.assertTrue(buf.is_full())
        self.assertEqual(buf.pop(), 42)
        self.assertTrue(buf.is_empty())

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            RingBuffer[int](0)
        with self.assertRaises(ValueError):
            RingBuffer[int](-5)

def demo() -> None:
    print("RingBuffer Demo")
    rb = RingBuffer[str](3)
    for word in ["hello", "world", "foo", "bar"]:
        try:
            rb.push(word)
            print(f"Pushed: {word}")
        except OverflowError:
            print(f"Buffer full, cannot push: {word}")
    print("Current buffer:", rb.to_list())
    while not rb.is_empty():
        print("Popped:", rb.pop())
    print("Buffer after popping all:", rb.to_list())

def benchmark() -> None:
    buf_size = 1000
    rb = RingBuffer[int](buf_size)
    def push_pop():
        for i in range(buf_size):
            rb.push(i)
        for _ in range(buf_size):
            rb.pop()
    time = timeit.timeit(push_pop, number=100)
    print(f"Push and pop {buf_size} items 100 times: {time:.4f} seconds")

if __name__ == "__main__":
    unittest.main(exit=False)
    demo()
    benchmark()
