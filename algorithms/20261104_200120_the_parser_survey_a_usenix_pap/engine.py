from __future__ import annotations
import re
from typing import List, Tuple, Iterable
from custom_types import Token, TokenType, Expr, Number, BinaryOp


class Lexer:
    """
    Simple lexical analyzer for arithmetic expressions.
    Supports integers, floats, +, -, *, /, parentheses and whitespace.
    """
    _token_specification: List[Tuple[str, str]] = [
        ("NUMBER", r"\d+(\.\d*)?|\.\d+"),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MUL", r"\*"),
        ("DIV", r"/"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("SKIP", r"[ \t\r\n]+"),
        ("MISMATCH", r"."),
    ]
    _tok_regex = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _token_specification))

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.tokens: List[Token] = []
        self._tokenize()

    def _tokenize(self) -> None:
        for mo in self._tok_regex.finditer(self.text):
            kind = mo.lastgroup
            value = mo.group()
            if kind == "NUMBER":
                token = Token(TokenType.NUMBER, float(value))
            elif kind == "PLUS":
                token = Token(TokenType.PLUS, value)
            elif kind == "MINUS":
                token = Token(TokenType.MINUS, value)
            elif kind == "MUL":
                token = Token(TokenType.MUL, value)
            elif kind == "DIV":
                token = Token(TokenType.DIV, value)
            elif kind == "LPAREN":
                token = Token(TokenType.LPAREN, value)
            elif kind == "RPAREN":
                token = Token(TokenType.RPAREN, value)
            elif kind == "SKIP":
                continue
            elif kind == "MISMATCH":
                raise SyntaxError(f"Unexpected character {value!r} at position {mo.start()}")
            else:
                raise RuntimeError(f"Unhandled token kind: {kind}")
            self.tokens.append(token)
        self.tokens.append(Token(TokenType.EOF, None))

    def __iter__(self) -> Iterable[Token]:
        return iter(self.tokens)


class Parser:
    """
    Recursive descent parser using the Shunting‑Yard algorithm to produce an AST.
    Grammar (with precedence):
        expr   ::= term ((PLUS|MINUS) term)*
        term   ::= factor ((MUL|DIV) factor)*
        factor ::= NUMBER | LPAREN expr RPAREN | (PLUS|MINUS) factor   # unary
    """
    # Operator precedence and associativity
    _precedence = {
        TokenType.PLUS: (1, "L"),
        TokenType.MINUS: (1, "L"),
        TokenType.MUL: (2, "L"),
        TokenType.DIV: (2, "L"),
    }

    def __init__(self, lexer: Lexer) -> None:
        self.tokens = list(lexer)
        self.pos = 0

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        current = self.tokens[self.pos]
        self.pos += 1
        return current

    def parse(self) -> Expr:
        """Parse the token stream into an AST."""
        output_queue: List[Token] = []
        operator_stack: List[Token] = []

        while True:
            token = self._peek()
            if token.type == TokenType.NUMBER:
                output_queue.append(self._advance())
            elif token.type in (TokenType.PLUS, TokenType.MINUS, TokenType.MUL, TokenType.DIV):
                while operator_stack:
                    top = operator_stack[-1]
                    if top.type == TokenType.LPAREN:
                        break
                    top_prec, _ = self._precedence[top.type]
                    cur_prec, cur_assoc = self._precedence[token.type]
                    if (cur_assoc == "L" and cur_prec <= top_prec) or (cur_assoc == "R" and cur_prec < top_prec):
                        output_queue.append(operator_stack.pop())
                    else:
                        break
                operator_stack.append(self._advance())
            elif token.type == TokenType.LPAREN:
                operator_stack.append(self._advance())
            elif token.type == TokenType.RPAREN:
                self._advance()  # consume RPAREN
                while operator_stack and operator_stack[-1].type != TokenType.LPAREN:
                    output_queue.append(operator_stack.pop())
                if not operator_stack:
                    raise SyntaxError("Mismatched parentheses")
                operator_stack.pop()  # discard the '('
            elif token.type == TokenType.EOF:
                self._advance()
                break
            else:
                raise SyntaxError(f"Unexpected token {token}")

        while operator_stack:
            op = operator_stack.pop()
            if op.type in (TokenType.LPAREN, TokenType.RPAREN):
                raise SyntaxError("Mismatched parentheses")
            output_queue.append(op)

        return self._build_ast_from_rpn(output_queue)

    def _build_ast_from_rpn(self, rpn: List[Token]) -> Expr:
        """Construct an AST from a list of tokens in Reverse Polish Notation."""
        stack: List[Expr] = []
        for token in rpn:
            if token.type == TokenType.NUMBER:
                stack.append(Number(token.value))  # type: ignore[arg-type]
            elif token.type in (TokenType.PLUS, TokenType.MINUS, TokenType.MUL, TokenType.DIV):
                if len(stack) < 2:
                    raise SyntaxError("Insufficient operands for binary operator")
                right = stack.pop()
                left = stack.pop()
                stack.append(BinaryOp(left, token.type, right))
            else:
                raise SyntaxError(f"Invalid token in RPN stream: {token}")
        if len(stack) != 1:
            raise SyntaxError("Invalid expression")
        return stack[0]


def evaluate(expr: Expr) -> float:
    """Recursively evaluate the AST."""
    if isinstance(expr, Number):
        return expr.value
    if isinstance(expr, BinaryOp):
        left_val = evaluate(expr.left)
        right_val = evaluate(expr.right)
        if expr.op == TokenType.PLUS:
            return left_val + right_val
        if expr.op == TokenType.MINUS:
            return left_val - right_val
        if expr.op == TokenType.MUL:
            return left_val * right_val
        if expr.op == TokenType.DIV:
            if right_val == 0:
                raise ZeroDivisionError("division by zero")
            return left_val / right_val
    raise TypeError(f"Unsupported expression node: {expr!r}")
