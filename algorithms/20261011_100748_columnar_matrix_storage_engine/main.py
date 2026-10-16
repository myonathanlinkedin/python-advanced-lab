import unittest
import time
from core import Matrix


class TestMatrix(unittest.TestCase):
    def test_creation_and_repr(self) -> None:
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.shape, (2, 2))
        self.assertEqual(repr(m), "Matrix([[1.0, 2.0], [3.0, 4.0]])")

    def test_factory_methods(self) -> None:
        z = Matrix.zeros(2, 3)
        self.assertEqual(z.shape, (2, 3))
        self.assertTrue(all(all(v == 0.0 for v in row) for row in z))

        o = Matrix.ones(1, 4)
        self.assertEqual(o.shape, (1, 4))
        self.assertTrue(all(v == 1.0 for v in o[0]))

        I = Matrix.identity(3)
        self.assertEqual(I.shape, (3, 3))
        for i in range(3):
            for j in range(3):
                self.assertEqual(I[i][j], 1.0 if i == j else 0.0)

    def test_elementwise_add_sub(self) -> None:
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        c = a + b
        self.assertEqual(c, Matrix([[6, 8], [10, 12]]))
        d = b - a
        self.assertEqual(d, Matrix([[4, 4], [4, 4]]))
        with self.assertRaises(ValueError):
            a + Matrix([[1, 2, 3]])

    def test_scalar_multiplication(self) -> None:
        m = Matrix([[1, -2], [3, 0]])
        self.assertEqual(3 * m, Matrix([[3, -6], [9, 0]]))
        self.assertEqual(m * -0.5, Matrix([[-0.5, 1.0], [-1.5, -0.0]]))

    def test_matrix_multiplication(self) -> None:
        a = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Matrix([[7, 8], [9, 10], [11, 12]])
        prod = a * b
        expected = Matrix([[58, 64], [139, 154]])
        self.assertEqual(prod, expected)
        with self.assertRaises(ValueError):
            b * a  # dimensions incompatible

    def test_transpose(self) -> None:
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        mt = m.transpose()
        self.assertEqual(mt, Matrix([[1, 4], [2, 5], [3, 6]]))

    def test_determinant(self) -> None:
        I = Matrix.identity(4)
        self.assertAlmostEqual(I.determinant(), 1.0)

        m = Matrix([[4, 3], [6, 3]])
        self.assertAlmostEqual(m.determinant(), -6.0)

        singular = Matrix([[2, 4], [1, 2]])
        self.assertAlmostEqual(singular.determinant(), 0.0)

        with self.assertRaises(ValueError):
            Matrix([[1, 2, 3], [4, 5, 6]]).determinant()

    def test_inverse(self) -> None:
        I = Matrix.identity(3)
        self.assertEqual(I.inverse(), I)

        m = Matrix([[4, 7], [2, 6]])
        inv = m.inverse()
        # Verify m * inv == I within tolerance
        prod = m * inv
        for i in range(2):
            for j in range(2):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(prod[i][j], expected, places=7)

        singular = Matrix([[1, 2], [2, 4]])
        with self.assertRaises(ValueError):
            singular.inverse()

    def test_error_handling(self) -> None:
        with self.assertRaises(ValueError):
            Matrix([])

        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3]])

        with self.assertRaises(ValueError):
            Matrix.zeros(0, 5)

        with self.assertRaises(TypeError):
            Matrix([[1, 2]]) * "not a matrix"

    def test_performance_small(self) -> None:
        # Simple benchmark to ensure deterministic runtime (no asserts on time)
        start = time.perf_counter()
        _ = Matrix.identity(50) * Matrix.identity(50)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 2.0)  # Should be fast on modest hardware


if __name__ == "__main__":
    unittest.main()
