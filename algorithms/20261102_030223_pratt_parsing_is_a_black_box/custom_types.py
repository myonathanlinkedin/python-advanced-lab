"""
Domain models for the Pratt Parser.

This module defines the core data structures:
1. Token: The atomic unit of lexical analysis.
2. ASTNode: The abstract syntax tree node representing parsed expressions.
3. Operator: Metadata for binary operators (precedence, associativity).

Design Invariants:
- Tokens are immutable value objects.
- ASTNodes form a tree structure with no cycles.
- Operator precedence is a positive integer (higher = binds tighter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union


class TokenType(Enum):
    """Enumeration of all token types in the language."""
    NUMBER = auto()
    IDENTIFIER = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    POWER = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    """
    Immutable token produced by the lexer.
    
    Attributes:
        type: The category of the token.
        value: The string representation (e.g., "42" for NUMBER, "+" for PLUS).
        position: The 0-based index in the source string where this token starts.
    """
    type: TokenType
    value: str
    position: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, '{self.value}', pos={self.position})"


class ASTNode:
    """
    Base class for all AST nodes.
    
    This is an abstract base class. Concrete implementations must override
    the `evaluate` method to provide semantic interpretation.
    """
    def evaluate(self, env: dict[str, float]) -> float:
        """
        Evaluate the expression represented by this AST node.
        
        Args:
            env: A mapping of variable names to their numeric values.
            
        Returns:
            The numeric result of the evaluation.
            
        Raises:
            ValueError: If a variable is not found in the environment.
            ZeroDivisionError: If division by zero is attempted.
        """
        raise NotImplementedError("Subclasses must implement evaluate()")


@dataclass
class NumberNode(ASTNode):
    """
    AST node representing a numeric literal.
    
    Attributes:
        value: The numeric value of the literal.
    """
    value: float

    def evaluate(self, env: dict[str, float]) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"NumberNode({self.value})"


@dataclass
class VariableNode(ASTNode):
    """
    AST node representing a variable reference.
    
    Attributes:
        name: The name of the variable.
    """
    name: str

    def evaluate(self, env: dict[str, float]) -> float:
        if self.name not in env:
            raise ValueError(f"Undefined variable: '{self.name}'")
        return env[self.name]

    def __repr__(self) -> str:
        return f"VariableNode({self.name})"


@dataclass
class BinaryOpNode(ASTNode):
    """
    AST node representing a binary operation.
    
    Attributes:
        op: The operator symbol (e.g., '+', '-', '*', '/', '**').
        left: The left operand AST node.
        right: The right operand AST node.
    """
    op: str
    left: ASTNode
    right: ASTNode

    def evaluate(self, env: dict[str, float]) -> float:
        left_val = self.left.evaluate(env)
        right_val = self.right.evaluate(env)
        
        if self.op == '+':
            return left_val + right_val
        elif self.op == '-':
            return left_val - right_val
        elif self.op == '*':
            return left_val * right_val
        elif self.op == '/':
            if right_val == 0:
                raise ZeroDivisionError("Division by zero")
            return left_val / right_val
        elif self.op == '**':
            return left_val ** right_val
        else:
            raise ValueError(f"Unknown operator: '{self.op}'")

    def __repr__(self) -> str:
        return f"BinaryOpNode({self.op}, {self.left}, {self.right})"


@dataclass
class UnaryOpNode(ASTNode):
    """
    AST node representing a unary operation (e.g., negation).
    
    Attributes:
        op: The operator symbol (e.g., '-').
        operand: The operand AST node.
    """
    op: str
    operand: ASTNode

    def evaluate(self, env: dict[str, float]) -> float:
        operand_val = self.operand.evaluate(env)
        if self.op == '-':
            return -operand_val
        else:
            raise ValueError(f"Unknown unary operator: '{self.op}'")

    def __repr__(self) -> str:
        return f"UnaryOpNode({self.op}, {self.operand})"


@dataclass
class Operator:
    """
    Metadata for a binary operator.
    
    Attributes:
        symbol: The string representation of the operator.
        precedence: The precedence level (higher binds tighter).
        associativity: 'left' or 'right' associativity.
    """
    symbol: str
    precedence: int
    associativity: str  # 'left' or 'right'

    def __repr__(self) -> str:
        return f"Operator({self.symbol}, prec={self.precedence}, assoc={self.associativity})"
