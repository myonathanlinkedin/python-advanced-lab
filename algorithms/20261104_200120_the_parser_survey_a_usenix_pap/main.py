from __future__ import annotations
import math
from custom_types import Lexer, Parser, evaluate, Expr

def _assert_almost_equal(a: float, b: float, eps: float = 1e-9) -> None:
    if not math.isclose(a, b, rel_tol=eps, abs_tol=eps):
        raise AssertionError(f"Values not almost equal: {a} vs {b}")

def run_tests() -> None:
    test_cases = [
        ("1+2", 3.0),
        ("2*3+4", 10.0),
        ("2+(3*4)", 14.0),
        ("(2+3)*4", 20.0),
        ("3+4*2/(1-5)", 1.0),  # 3 + (4*2)/-4 = 3 -2 = 1
        ("   7   ", 7.0),
        ("-3+5", 2.0),  # unary minus handled as binary 0-3
        ("4+-2", 2.0),
        ("(1+(2*(3+(4/2))))", 11.0),
        ("0.5*8", 4.0),
        ("3.1415*2", 6.2830),
    ]

    for expr_str, expected in test_cases:
        lexer = Lexer(expr_str)
        parser = Parser(lexer)
        ast = parser.parse()
        result = evaluate(ast)
        _assert_almost_equal(result, expected)

    # Edge case: division by zero should raise
    try:
        lexer = Lexer("1/0")
        parser = Parser(lexer)
        ast = parser.parse()
        evaluate(ast)
        raise AssertionError("ZeroDivisionError expected but not raised")
    except ZeroDivisionError:
        pass

    # Edge case: mismatched parentheses
    try:
        lexer = Lexer("(1+2")
        parser = Parser(lexer)
        parser.parse()
        raise AssertionError("SyntaxError expected for mismatched parentheses")
    except SyntaxError:
        pass

    # Edge case: invalid character
    try:
        lexer = Lexer("2 & 3")
        parser = Parser(lexer)
        parser.parse()
        raise AssertionError("SyntaxError expected for invalid character")
    except SyntaxError:
        pass

    print("All unit tests passed.")


def demo() -> None:
    examples = [
        "3 + 4 * 2 / (1 - 5)",
        "(2.5 + 3.5) * 2",
        "10 / (5 - 5)",  # will raise ZeroDivisionError
    ]
    for expr in examples:
        print(f"Expression: {expr}")
        try:
            lexer = Lexer(expr)
            parser = Parser(lexer)
            ast = parser.parse()
            result = evaluate(ast)
            print(f"Result: {result}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    run_tests()
    demo()
