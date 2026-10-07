import unittest
import timeit
from core import Tensor

class TestTensor(unittest.TestCase):
    def test_creation_from_nested(self):
        t = Tensor(data=[[1, 2], [3, 4]])
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.size, 4)
        self.assertEqual(t[0, 1], 2)

    def test_creation_with_shape_and_fill(self):
        t = Tensor(shape=(3, 2), fill=7)
        self.assertEqual(t.shape, (3, 2))
        self.assertEqual(t.tolist(), [[7, 7], [7, 7], [7, 7]])

    def test_indexing_and_assignment(self):
        t = Tensor(shape=(2, 3), fill=0)
        t[1, 2] = 5
        self.assertEqual(t[1, 2], 5)
        with self.assertRaises(TypeError):
            t[0:1, :] = Tensor(shape=(1, 3), fill=1)  # disallowed slice assignment

    def test_slice_returns_subtensor(self):
        t = Tensor(data=[[1, 2, 3], [4, 5, 6]])
        sub = t[0, 1:]          # slice on second axis
        self.assertIsInstance(sub, Tensor)
        self.assertEqual(sub.shape, (1, 2))
        self.assertEqual(sub.tolist(), [[2, 3]])

    def test_reshape(self):
        t = Tensor(data=[[1, 2], [3, 4]])
        r = t.reshape((4,))
        self.assertEqual(r.shape, (4,))
        self.assertEqual(r.tolist(), [1, 2, 3, 4])

    def test_transpose(self):
        t = Tensor(data=[[1, 2, 3], [4, 5, 6]])
        tr = t.transpose()
        self.assertEqual(tr.shape, (3, 2))
        self.assertEqual(tr.tolist(), [[1, 4], [2, 5], [3, 6]])

    def test_elementwise_addition(self):
        a = Tensor(data=[[1, 2], [3, 4]])
        b = Tensor(shape=(2, 2), fill=10)
        c = a + b
        self.assertEqual(c.tolist(), [[11, 12], [13, 14]])

    def test_scalar_broadcast(self):
        a = Tensor(shape=(2, 2), fill=3)
        b = a * 2
        self.assertEqual(b.tolist(), [[6, 6], [6, 6]])

    def test_map(self):
        t = Tensor(data=[[1, 2], [3, 4]])
        s = t.map(lambda x: x * x)
        self.assertEqual(s.tolist(), [[1, 4], [9, 16]])

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            Tensor(shape=(0, 2))

    def test_mismatched_shape_on_op(self):
        a = Tensor(shape=(2, 2), fill=1)
        b = Tensor(shape=(2, 3), fill=1)
        with self.assertRaises(ValueError):
            _ = a + b

def benchmark():
    setup = "from core import Tensor; t = Tensor(shape=(100,100), fill=1)"
    stmt = "t + t"
    time = timeit.timeit(stmt, setup=setup, number=100)
    print(f"Benchmark (100 adds of 100x100 tensors): {time:.4f}s")

if __name__ == '__main__':
    # Run unit tests
    unittest.main(exit=False)

    # Simple benchmark demonstration
    benchmark()
