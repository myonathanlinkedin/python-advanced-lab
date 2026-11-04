from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

# Token definitions
@dataclass(frozen=True)
class Token:
    type: str
    value: str

# Simple lexer for arithmetic expressions
class Lexer:
    _ops = set('+-*/^')
    _whitespace = set(' \t\r\n')

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if self.text else None

    def _advance(self) -> None:
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def _skip_ws(self) -> None:
        while self.current_char and self.current_char in self._whitespace:
            self._advance()

    def _number(self) -> Token:
        num_chars = []
        dot_seen = False
        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if dot_seen:
                    break
                dot_seen = True
            num_chars.append(self.current_char)
            self._advance()
        return Token('NUMBER', ''.join(num_chars))

    def generate_tokens(self) -> List[Token]:
        tokens: List[Token] = []
        while self.current_char is not None:
            if self.current_char in self._whitespace:
                self._skip_ws()
                continue
            if self.current_char.isdigit() or self.current_char == '.':
                tokens.append(self._number())
                continue
            if self.current_char in self._ops:
                tokens.append(Token('OP', self.current_char))
                self._advance()
                continue
            if self.current_char == '(':
                tokens.append(Token('LPAREN', '('))
                self._advance()
                continue
            if self.current_char == ')':
                tokens.append(Token('RPAREN', ')'))
                self._advance()
                continue
            raise SyntaxError(f'Invalid character: {self.current_char}')
        tokens.append(Token('EOF', ''))
        return tokens

# AST node definitions
class ASTNode:
    pass

@dataclass
class NumberNode(ASTNode):
    value: float

@dataclass
class UnaryOpNode(ASTNode):
    op: str
    operand: ASTNode

@dataclass
class BinaryOpNode(ASTNode):
    left: ASTNode
    op: str
    right: ASTNode

# Pratt parser implementation
class PrattParser:
    # binding powers: (left binding power, right binding power)
    _bp: Dict[str, Tuple[int, int]] = {
        '+': (10, 10),
        '-': (10, 10),
        '*': (20, 20),
        '/': (20, 20),
        '^': (30, 29),   # right‑associative
    }

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.tok = self.tokens[self.pos]

    def _advance(self) -> None:
        self.pos += 1
        if self.pos < len(self.tokens):
            self.tok = self.tokens[self.pos]
        else:
            self.tok = Token('EOF', '')

    def parse(self) -> ASTNode:
        expr = self._expression(0)
        if self.tok.type != 'EOF':
            raise SyntaxError('Unexpected token after expression')
        return expr

    # nud: null denotation
    def _nud(self, token: Token) -> ASTNode:
        if token.type == 'NUMBER':
            self._advance()
            return NumberNode(float(token.value))
        if token.type == 'OP' and token.value in '+-':
            # unary plus/minus
            self._advance()
            right = self._expression(self._bp['+'][0])  # same precedence as binary +/-
            return UnaryOpNode(token.value, right)
        if token.type == 'LPAREN':
            self._advance()
            expr = self._expression(0)
            if self.tok.type != 'RPAREN':
                raise SyntaxError('Missing closing parenthesis')
            self._advance()
            return expr
        raise SyntaxError(f'Unexpected token: {token}')

    # led: left denotation
    def _led(self, left: ASTNode, token: Token) -> ASTNode:
        if token.type != 'OP' or token.value not in self._bp:
            raise SyntaxError(f'Invalid infix operator: {token}')
        lbp, rbp = self._bp[token.value]
        self._advance()
        right = self._expression(rbp)
        return BinaryOpNode(left, token.value, right)

    def _expression(self, rbp: int) -> ASTNode:
        t = self.tok
        self._advance()
        left = self._nud(t)
        while rbp < self._lbp():
            t = self.tok
            self._advance()
            left = self._led(left, t)
        return left

    def _lbp(self) -> int:
        if self.tok.type == 'OP' and self.tok.value in self._bp:
            return self._bp[self.tok.value][0]
        if self.tok.type in ('EOF', 'RPAREN'):
            return 0
        return 0

# Helper to parse a string into an AST
def parse_expression(text: str) -> ASTNode:
    lexer = Lexer(text)
    tokens = lexer.generate_tokens()
    parser = PrattParser(tokens)
    return parser.parse()

# Simple evaluator for the AST
def evaluate(node: ASTNode) -> float:
    if isinstance(node, NumberNode):
        return node.value
    if isinstance(node, UnaryOpNode):
        val = evaluate(node.operand)
        return +val if node.op == '+' else -val
    if isinstance(node, BinaryOpNode):
        left = evaluate(node.left)
        right = evaluate(node.right)
        if node.op == '+':
            return left + right
        if node.op == '-':
            return left - right
        if node.op == '*':
            return left * right
        if node.op == '/':
            return left / right
        if node.op == '^':
            return left ** right
    raise TypeError(f'Unsupported AST node: {node}')
