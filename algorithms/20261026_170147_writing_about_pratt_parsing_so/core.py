"""Pratt parsing (top-down operator precedence) for a small arithmetic expression language.

Grammar (informal):
    expr    := term
    term    := factor ( ('+' | '-') factor )*
    factor  := primary ( ('*' | '/') primary )*
    primary := NUMBER | '(' expr ')' | '-' primary

Pratt's insight: give every token a *binding power* (precedence) and let the
parser recursively descend only while the next token binds tighter than the
current precedence.  This yields a single, uniform loop that handles both
infix and prefix operators without a separate precedence table per level.

Complexity:
    Time   O(n)  -- each token is consumed exactly once.
    Space  O(d)  -- recursion depth equals the maximum nesting of parentheses
                    (worst case O(n) for fully nested input).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Num:
    """Literal numeric leaf."""
    value: float


@dataclass(frozen=True)
class BinOp:
    """Binary operation node."""
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Neg:
    """Unary minus (prefix negation)."""
    operand: "Expr"


Expr = Union[Num, BinOp, Neg]


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Token:
    kind: str          # 'NUM', 'PLUS', 'MINUS', 'STAR', 'SLASH', 'LPAREN', 'RPAREN', 'EOF'
    value: str = ""    # raw text (for NUM this is the digit string)


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------
def tokenize(src: str) -> List[Token]:
    """Convert a source string into a flat list of tokens.

    Raises ValueError on illegal characters.
    Time O(n), Space O(n).
    """
    tokens: List[Token] = []
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if ch.isspace():
            i += 1
        elif ch.isdigit() or ch == '.':
            j = i
            seen_dot = False
            while j < n and (src[j].isdigit() or (src[j] == '.' and not seen_dot)):
                if src[j] == '.':
                    seen_dot = True
                j += 1
            tokens.append(Token('NUM', src[i:j]))
            i = j
        elif ch == '+':
            tokens.append(Token('PLUS'))
            i += 1
        elif ch == '-':
            tokens.append(Token('MINUS'))
            i += 1
        elif ch == '*':
            tokens.append(Token('STAR'))
            i += 1
        elif ch == '/':
            tokens.append(Token('SLASH'))
            i += 1
        elif ch == '(':
            tokens.append(Token('LPAREN'))
            i += 1
        elif ch == ')':
            tokens.append(Token('RPAREN'))
            i += 1
        else:
            raise ValueError(f"Illegal character {ch!r} at position {i}")
    tokens.append(Token('EOF'))
    return tokens


# ---------------------------------------------------------------------------
# Pratt Parser
# ---------------------------------------------------------------------------
# Binding powers (higher = tighter).  Chosen so that:
#   * and /  bind tighter than + and -
#   unary -  binds tighter than * and /
#   parentheses  always force a new sub-expression
BP_EOF      = 0
BP_ADD      = 10   # + -
BP_MUL      = 20   # * /
BP_NEG      = 25   # unary -
BP_ATOM     = 30   # atoms (numbers, parenthesized exprs)


class Parser:
    """Recursive-descent Pratt parser.

    The single entry point is ``parse_expression(min_bp)``.
    """

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # -- helpers ----------------------------------------------------------
    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, kind: str) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            raise SyntaxError(f"Expected {kind}, got {tok.kind} ({tok.value!r})")
        return self._advance()

    # -- Pratt core -------------------------------------------------------
    def parse_expression(self, min_bp: int = 0) -> Expr:
        """Parse an expression with binding power >= *min_bp*.

        This is the heart of Pratt parsing:
          1. Parse the left-hand "nud" (null denotation) -- the atom or prefix op.
          2. While the next infix token's binding power exceeds *min_bp*,
             consume it and recursively parse the right operand with
             *left_bp + 1* (the +1 enforces left-associativity).
        """
        left = self._nud()

        while True:
            tok = self._peek()
            bp = self._infix_bp(tok)
            if bp <= min_bp:
                break
            self._advance()
            right = self.parse_expression(bp + 1)
            left = BinOp(tok.kind, left, right)

        return left

    def _nud(self) -> Expr:
        """Null denotation: parse an atom or prefix operator."""
        tok = self._peek()
        if tok.kind == 'NUM':
            self._advance()
            return Num(float(tok.value))
        if tok.kind == 'LPAREN':
            self._advance()
            expr = self.parse_expression(0)
            self._expect('RPAREN')
            return expr
        if tok.kind == 'MINUS':
            self._advance()
            operand = self.parse_expression(BP_NEG)
            return Neg(operand)
        raise SyntaxError(f"Unexpected token {tok.kind} ({tok.value!r}) in nud")

    @staticmethod
    def _infix_bp(tok: Token) -> int:
        """Return the binding power of an infix token, or 0 if not infix."""
        if tok.kind in ('PLUS', 'MINUS'):
            return BP_ADD
        if tok.kind in ('STAR', 'SLASH'):
            return BP_MUL
        return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse(src: str) -> Expr:
    """Tokenize and parse *src* into an AST.  Raises SyntaxError on bad input."""
    tokens = tokenize(src)
    parser = Parser(tokens)
    expr = parser.parse_expression(0)
    if parser._peek().kind != 'EOF':
        raise SyntaxError(f"Trailing tokens after expression: {parser._peek()}")
    return expr


# ---------------------------------------------------------------------------
# Evaluator (for testing / demo)
# ---------------------------------------------------------------------------
def evaluate(node: Expr) -> float:
    """Recursively evaluate an AST.  Time O(n), Space O(d)."""
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Neg):
        return -evaluate(node.operand)
    if isinstance(node, BinOp):
        l = evaluate(node.left)
        r = evaluate(node.right)
        if node.op == 'PLUS':
            return l + r
        if node.op == 'MINUS':
            return l - r
        if node.op == 'STAR':
            return l * r
        if node.op == 'SLASH':
            if r == 0:
                raise ZeroDivisionError("Division by zero")
            return l / r
    raise TypeError(f"Unknown node type: {type(node)}")


# ---------------------------------------------------------------------------
# Pretty-printer (for demo / debugging)
# ---------------------------------------------------------------------------
def to_string(node: Expr) -> str:
    """Render an AST back to a readable infix string."""
    if isinstance(node, Num):
        v = node.value
        if v == int(v):
            return str(int(v))
        return str(v)
    if isinstance(node, Neg):
        inner = to_string(node.operand)
        if isinstance(node.operand, BinOp):
            return f"(-{inner})"
        return f"-{inner}"
    if isinstance(node, BinOp):
        l = to_string(node.left)
        r = to_string(node.right)
        # Add parens around children that have lower precedence
        child_bp = _node_bp(node.left)
        parent_bp = _op_bp(node.op)
        l_str = f"({l})" if child_bp < parent_bp else l
        child_bp_r = _node_bp(node.right)
        r_str = f"({r})" if child_bp_r < parent_bp else r
        return f"{l_str} {node.op.lower()} {r_str}"
    raise TypeError(f"Unknown node type: {type(node)}")


def _op_bp(op: str) -> int:
    if op in ('PLUS', 'MINUS'):
        return BP_ADD
    if op in ('STAR', 'SLASH'):
        return BP_MUL
    return 0


def _node_bp(node: Expr) -> int:
    if isinstance(node, BinOp):
        return _op_bp(node.op)
    if isinstance(node, Neg):
        return BP_NEG
    return BP_ATOM
