import unittest
from core import parse_expression, evaluate, NumberNode, UnaryOpNode, BinaryOpNode

class TestPrattParser(unittest.TestCase):
    def assertEval(self, expr: str, expected: float) -> None:
        ast = parse_expression(expr)
        result = evaluate(ast)
        self.assertAlmostEqual(result, expected, places=7, msg=f'Expr: {expr}')

    def test_number(self):
        self.assertEval('42', 42)
        self.assertEval('3.14', 3.14)

    def test_simple_binary(self):
        self.assertEval('1+2', 3)
        self.assertEval('4-5', -1)
        self.assertEval('6*7', 42)
        self.assertEval('8/2', 4)

    def test_precedence(self):
        self.assertEval('1+2*3', 7)
        self.assertEval('4*5+6', 26)
        self.assertEval('2+3*4^2', 2 + 3 * (4 ** 2))

    def test_associativity(self):
        self.assertEval('2^3^2', 2 ** (3 ** 2))   # right‑associative
        self.assertEval('10-5-2', (10 - 5) - 2)   # left‑associative

    def test_parentheses(self):
        self.assertEval('(1+2)*3', 9)
        self.assertEval('2*(3+4*(5-6))', 2 * (3 + 4 * (5 - 6)))

    def test_unary(self):
        self.assertEval('-5', -5)
        self.assertEval('+7', 7)
        self.assertEval('-3+4', 1)
        self.assertEval('-(2+3)*4', -20)

    def test_complex(self):
        expr = '-(1+2)*(3^2-4/2)+5'
        expected = -(1+2)*(3**2-4/2)+5
        self.assertEval(expr, expected)

    def test_invalid(self):
        with self.assertRaises(SyntaxError):
            parse_expression('1++2')
        with self.assertRaises(SyntaxError):
            parse_expression('(')
        with self.assertRaises(SyntaxError):
            parse_expression('2 $ 3')

    def test_ast_structure(self):
        ast = parse_expression('1+2*3')
        self.assertIsInstance(ast, BinaryOpNode)
        self.assertEqual(ast.op, '+')
        self.assertIsInstance(ast.right, BinaryOpNode)
        self.assertEqual(ast.right.op, '*')
        self.assertIsInstance(ast.right.left, NumberNode)
        self.assertEqual(ast.right.left.value, 2.0)

if __name__ == '__main__':
    unittest.main()
