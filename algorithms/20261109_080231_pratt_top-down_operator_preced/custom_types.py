"""
Domain models for the Pratt Top-Down Operator Precedence Expression Parser.

This module defines the core data structures:
- Token: Lexical unit with type, value, and position.
- TokenType: Enumeration of all token categories.
- ASTNode: Abstract base class for all expression tree nodes.
- Concrete AST node classes: Number, Variable, UnaryOp, BinaryOp, Call.
- ParseError: Custom exception for parse failures.

Time/Space Invariants:
- Token creation: O(1) time, O(1) space.
- AST node creation: O(1) time, O(1) space per node.
- Total AST size: O(n) where n is the number of tokens.
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Union


class TokenType(Enum):
    """Enumeration of all token types produced by the lexer."""
    NUMBER = auto()
    IDENTIFIER = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    POWER = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    """
    Immutable lexical token.

    Attributes:
        type: The category of the token.
        value: The string representation (for NUMBER and IDENTIFIER).
        pos: Zero-based column index in the source string.
    """
    type: TokenType
    value: str
    pos: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, pos={self.pos})"


class ParseError(Exception):
    """Raised when the parser encounters an unrecoverable syntax error."""
    def __init__(self, message: str, pos: int = -1) -> None:
        self.pos = pos
        super().__init__(f"Parse error at position {pos}: {message}")


# --- AST Node Hierarchy ---

class ASTNode:
    """Abstract base class for all expression tree nodes."""
    def __repr__(self) -> str:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Number(ASTNode):
    """Literal numeric constant."""
    value: float

    def __repr__(self) -> str:
        if self.value == int(self.value):
            return f"Number({int(self.value)})"
        return f"Number({self.value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Number):
            return NotImplemented
        return self.value == other.value


@dataclass(frozen=True)
class Variable(ASTNode):
    """Named variable reference."""
    name: str

    def __repr__(self) -> str:
        return f"Variable({self.name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Variable):
            return NotImplemented
        return self.name == other.name


@dataclass(frozen=True)
class UnaryOp(ASTNode):
    """
    Unary operator application.

    Attributes:
        op: The operator symbol ('+', '-', '~', '!').
        operand: The sub-expression.
    """
    op: str
    operand: ASTNode

    def __repr__(self) -> str:
        return f"UnaryOp({self.op!r}, {self.operand!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnaryOp):
            return NotImplemented
        return self.op == other.op and self.operand == other.operand


@dataclass(frozen=True)
class BinaryOp(ASTNode):
    """
    Binary operator application.

    Attributes:
        op: The operator symbol ('+', '-', '*', '/', '%', '**').
        left: Left sub-expression.
        right: Right sub-expression.
    """
    op: str
    left: ASTNode
    right: ASTNode

    def __repr__(self) -> str:
        return f"BinaryOp({self.op!r}, {self.left!r}, {self.right!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinaryOp):
            return NotImplemented
        return (self.op == other.op and
                self.left == other.left and
                self.right == other.right)


@dataclass(frozen=True)
class Call(ASTNode):
    """
    Function call expression.

    Attributes:
        func: The function name.
        args: List of argument expressions.
    """
    func: str
    args: tuple  # tuple for hashability/immutability

    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.args)
        return f"Call({self.func!r}, [{args_str}])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Call):
            return NotImplemented
        return self.func == other.func and self.args == other.args


# Type alias for any AST node
Expr = Union[Number, Variable, UnaryOp, BinaryOp, Call]
