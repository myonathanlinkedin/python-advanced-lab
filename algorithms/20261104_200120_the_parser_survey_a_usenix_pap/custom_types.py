from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union, List, Optional


class TokenType(Enum):
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: Optional[Union[float, str]] = None

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value})"


class Expr:
    """Base class for all expression AST nodes."""
    pass


@dataclass
class Number(Expr):
    value: float

    def __repr__(self) -> str:
        return f"Number({self.value})"


@dataclass
class BinaryOp(Expr):
    left: Expr
    op: TokenType
    right: Expr

    def __repr__(self) -> str:
        return f"BinaryOp({self.left}, {self.op.name}, {self.right})"
