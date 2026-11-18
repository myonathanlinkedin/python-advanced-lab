import unittest
from typing import List

# Import from core
from core import (
    SexpMatrix,
    evaluate_sexp,
    parse_sexp,
    sexp_to_str,
)

class TestSexpParsing(unittest.TestCase):
    def test_atom_parsing(self):
        self.assertEqual(parse_sexp("42"), 42)
        self.assertAlmostEqual(parse_sexp("3.14"), 3.14)
        self.assertEqual(parse_sexp("x"), "x")

    def test_nested_parsing(self):
        expr = parse_sexp("(+ 1 (* x 2))")
        self.assertEqual(expr, ['+', 1, ['*', 'x', 2]])

    def test_invalid_syntax(self):
        with self.assertRaises(SyntaxError):
            parse_sexp("(+ 1 2")
        with self.assertRaises(SyntaxError):
            parse_sexp(") 1 2 (")

class TestSexpEvaluation(unittest.TestCase):
    def test_simple_arithmetic(self):
        expr = parse_sexp("(+ 3 4)")
        self.assertEqual(evaluate_sexp(expr, {}), 7)

    def test_variables(self):
        expr = parse_sexp("(* x y)")
        env = {'x': 5, 'y': 2}
        self.assertEqual(evaluate_sexp(expr, env), 10)

    def test_unbound_variable(self):
        expr = parse_sexp("(* x 2)")
        with self.assertRaises(NameError):
            evaluate_sexp(expr, {})

    def test_unsupported_operator(self):
        expr = parse_sexp("(max 1 2)")
        with self.assertRaises(ValueError):
            evaluate_sexp(expr, {})

class TestSexpMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.A = SexpMatrix([
            [parse_sexp("x"), parse_sexp("1")],
            [parse_sexp("2"), parse_sexp("y")]
        ])
        self.B = SexpMatrix([
            [parse_sexp("3"), parse_sexp("z")],
            [parse_sexp("4"), parse_sexp("5")]
        ])

    def test_shape(self):
        self.assertEqual(self.A.shape, (2, 2))
        self.assertEqual(self.B.shape, (2, 2))

    def test_element_access(self):
        self.assertEqual(self.A[0, 0], "x")
        self.assertEqual(self.B[1, 1], 5)

    def test_addition(self):
        C = self.A + self.B
        expected = [
            [['+', 'x', 3], ['+', 1, 'z']],
            [['+', 2, 4], ['+', 'y', 5]]
        ]
        self.assertEqual(C._data, expected)

    def test_multiplication(self):
        C = self.A @ self.B
        # Manually compute symbolic result for verification
        expected = [
            [
                ['+', ['*', 'x', 3], ['*', 1, 4]],
                ['+', ['*', 'x', 'z'], ['*', 1, 5]]
            ],
            [
                ['+', ['*', 2, 3], ['*', 'y', 4]],
                ['+', ['*', 2, 'z'], ['*', 'y', 5]]
            ]
        ]
        self.assertEqual(C._data, expected)

    def test_evaluation(self):
        C = self.A @ self.B
        env = {'x': 1, 'y': 2, 'z': 3}
        numeric = C.evaluate(env)
        expected = [
            [1*3 + 1*4, 1*3 + 1*5],
            [2*3 + 2*4, 2*3 + 2*5]
        ]
        self.assertEqual(numeric, expected)

    def test_dimension_mismatch_add(self):
        D = SexpMatrix([[1, 2, 3]])
        with self.assertRaises(ValueError):
            _ = self.A + D

    def test_dimension_mismatch_mul(self):
        D = SexpMatrix([[1], [2], [3]])  # 3x1
        with self.assertRaises(ValueError):
            _ = self.A @ D

class TestUtilityFunctions(unittest.TestCase):
    def test_sexp_to_str(self):
        expr = parse_sexp("(+ (* a b) 3)")
        s = sexp_to_str(expr)
        self.assertEqual(s, "(+ (* a b) 3)")

if __name__ == '__main__':
    unittest.main()
