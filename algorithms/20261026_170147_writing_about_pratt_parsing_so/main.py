"""Unit tests, benchmarks, and demo entry-point for the Pratt parser."""
from __future__ import annotations

import time
import unittest

from core import (
    BinOp, Expr, Neg, Num,
    evaluate, parse, to_string, tokenize,
)


class TestTokenizer(unittest.TestCase):
    def test_simple(self):
        toks = tokenize("1 + 2")
        self.assertEqual([t.kind for t in toks], ['NUM', 'PLUS', 'NUM', 'EOF'])

    def test_float(self):
        toks = tokenize("3.14")
        self.assertEqual(toks[0].value, "3.14")

    def test_illegal_char(self):
        with self.assertRaises(ValueError):
            tokenize("1 & 2")

    def test_empty(self):
        toks = tokenize("")
        self.assertEqual(toks[0].kind, 'EOF')


class TestParser(unittest.TestCase):
    def test_number(self):
        ast = parse("42")
        self.assertIsInstance(ast, Num)
        self.assertEqual(ast.value, 42.0)

    def test_addition(self):
        ast = parse("1 + 2")
        self.assertIsInstance(ast, BinOp)
        self.assertEqual(ast.op, 'PLUS')
        self.assertEqual(evaluate(ast), 3.0)

    def test_precedence(self):
        """2 + 3 * 4 should be 2 + (3*4) = 14, not (2+3)*4 = 20."""
        ast = parse("2 + 3 * 4")
        self.assertEqual(evaluate(ast), 14.0)

    def test_left_associativity(self):
        """10 - 3 - 2 should be (10-3)-2 = 5, not 10-(3-2) = 9."""
        ast = parse("10 - 3 - 2")
        self.assertEqual(evaluate(ast), 5.0)

    def test_division_left_assoc(self):
        """100 / 10 / 2 = (100/10)/2 = 5."""
        ast = parse("100 / 10 / 2")
        self.assertEqual(evaluate(ast), 5.0)

    def test_parentheses(self):
        ast = parse("(1 + 2) * 3")
        self.assertEqual(evaluate(ast), 9.0)

    def test_nested_parens(self):
        ast = parse("((1 + 2) * (3 + 4))")
        self.assertEqual(evaluate(ast), 21.0)

    def test_unary_minus(self):
        ast = parse("-5")
        self.assertIsInstance(ast, Neg)
        self.assertEqual(evaluate(ast), -5.0)

    def test_unary_minus_precedence(self):
        """-2 * 3 should be (-2)*3 = -6."""
        ast = parse("-2 * 3")
        self.assertEqual(evaluate(ast), -6.0)

    def test_unary_minus_with_parens(self):
        ast = parse("-(1 + 2)")
        self.assertEqual(evaluate(ast), -3.0)

    def test_complex_expression(self):
        """1 + 2 * 3 - 4 / 2 = 1 + 6 - 2 = 5."""
        ast = parse("1 + 2 * 3 - 4 / 2")
        self.assertEqual(evaluate(ast), 5.0)

    def test_deeply_nested(self):
        src = "1" + " + 1" * 50
        ast = parse(src)
        self.assertEqual(evaluate(ast), 51.0)

    def test_trailing_tokens(self):
        with self.assertRaises(SyntaxError):
            parse("1 + 2 3")

    def test_missing_rparen(self):
        with self.assertRaises(SyntaxError):
            parse("(1 + 2")

    def test_empty_expression(self):
        with self.assertRaises(SyntaxError):
            parse("")

    def test_division_by_zero(self):
        ast = parse("1 / 0")
        with self.assertRaises(ZeroDivisionError):
            evaluate(ast)


class TestToString(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(to_string(parse("1 + 2")), "1 + 2")

    def test_precedence_parens(self):
        """(1+2)*3 should render with parens around 1+2."""
        s = to_string(parse("(1 + 2) * 3"))
        self.assertEqual(s, "(1 + 2) * 3")

    def test_no_unnecessary_parens(self):
        s = to_string(parse("1 + 2 * 3"))
        self.assertEqual(s, "1 + 2 * 3")

    def test_unary(self):
        s = to_string(parse("-5"))
        self.assertEqual(s, "-5")


class TestBenchmark(unittest.TestCase):
    def test_large_expression(self):
        """Parse and evaluate a 10 000-term expression in reasonable time."""
        n = 10_000
        src = " + ".join(["1"] * n)
        t0 = time.perf_counter()
        ast = parse(src)
        t1 = time.perf_counter()
        result = evaluate(ast)
        t2 = time.perf_counter()
        self.assertEqual(result, float(n))
        parse_ms = (t1 - t0) * 1000
        eval_ms = (t2 - t1) * 1000
        print(f"\n[Benchmark] n={n}: parse={parse_ms:.1f}ms  eval={eval_ms:.1f}ms")
        # Should complete well under 2 seconds
        self.assertLess(parse_ms + eval_ms, 2000)


def _demo() -> None:
    """Interactive-style demo showing parse tree and evaluation."""
    examples = [
        "1 + 2 * 3",
        "(1 + 2) * 3",
        "10 - 3 - 2",
        "-2 * 3 + 1",
        "((1 + 2) * (3 + 4)) / 2",
        "100 / 10 / 2",
    ]
    print("=" * 50)
    print("  Pratt Parser Demo")
    print("=" * 50)
    for src in examples:
        ast = parse(src)
        result = evaluate(ast)
        rendered = to_string(ast)
        print(f"  {src:<25s}  =>  {result:<10.4g}  AST: {rendered}")
    print("=" * 50)


if __name__ == '__main__':
    _demo()
    unittest.main(verbosity=2)
