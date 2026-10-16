import re
from dataclasses import dataclass
from typing import List, Iterator, Optional, Union

# Token definitions
@dataclass(frozen=True)
class Token:
    type: str
    value: str

# Token types
NUMBER      = "NUMBER"
IDENTIFIER  = "IDENTIFIER"
PLUS        = "PLUS"
MINUS       = "MINUS"
TIMES       = "TIMES"
DIVIDE      = "DIVIDE"
LPAREN      = "LPAREN"
RPAREN      = "RPAREN"
EOF         = "EOF"

# Lexer using a finite state machine
class Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.current_char: Optional[str] = self.text[self.pos] if self.text else None

    def advance(self) -> None:
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def skip_whitespace(self) -> None:
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self) -> Token:
        result = ""
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            result += self.current_char
            self.advance()
        return Token(NUMBER, result)

    def identifier(self) -> Token:
        result = ""
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        return Token(IDENTIFIER, result)

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            if self.current_char.isdigit():
                tokens.append(self.number())
                continue
            if self.current_char.isalpha() or self.current_char == '_':
                tokens.append(self.identifier())
                continue
            if self.current_char == '+':
                tokens.append(Token(PLUS, '+'))
                self.advance()
                continue
            if self.current_char == '-':
                tokens.append(Token(MINUS, '-'))
                self.advance()
                continue
            if self.current_char == '*':
                tokens.append(Token(TIMES, '*'))
                self.advance()
                continue
            if self.current_char == '/':
                tokens.append(Token(DIVIDE, '/'))
                self.advance()
                continue
            if self.current_char == '(':
                tokens.append(Token(LPAREN, '('))
                self.advance()
                continue
            if self.current_char == ')':
                tokens.append(Token(RPAREN, ')'))
                self.advance()
                continue
            raise ValueError(f"Unexpected character: {self.current_char}")
        tokens.append(Token(EOF, ''))
        return tokens

# AST node definitions
@dataclass(frozen=True)
class Number:
    value: float

@dataclass(frozen=True)
class Identifier:
    name: str

@dataclass(frozen=True)
class BinOp:
    left: Union['Number', 'Identifier', 'BinOp']
    op: str
    right: Union['Number', 'Identifier', 'BinOp']

# Parser using recursive descent
class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.current_token: Token = self.tokens[self.pos]

    def eat(self, token_type: str) -> None:
        if self.current_token.type == token_type:
            self.pos += 1
            self.current_token = self.tokens[self.pos]
        else:
            raise ValueError(f"Expected token {token_type}, got {self.current_token.type}")

    def factor(self) -> Union[Number, Identifier, BinOp]:
        token = self.current_token
        if token.type == NUMBER:
            self.eat(NUMBER)
            return Number(float(token.value))
        if token.type == IDENTIFIER:
            self.eat(IDENTIFIER)
            return Identifier(token.value)
        if token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node
        if token.type == MINUS:
            self.eat(MINUS)
            node = self.factor()
            return BinOp(Number(0), '-', node)
        raise ValueError(f"Unexpected token {token.type}")

    def term(self) -> Union[Number, Identifier, BinOp]:
        node = self.factor()
        while self.current_token.type in (TIMES, DIVIDE):
            op_token = self.current_token
            if op_token.type == TIMES:
                self.eat(TIMES)
            else:
                self.eat(DIVIDE)
            node = BinOp(node, op_token.value, self.factor())
        return node

    def expr(self) -> Union[Number, Identifier, BinOp]:
        node = self.term()
        while self.current_token.type in (PLUS, MINUS):
            op_token = self.current_token
            if op_token.type == PLUS:
                self.eat(PLUS)
            else:
                self.eat(MINUS)
            node = BinOp(node, op_token.value, self.term())
        return node

    def parse(self) -> Union[Number, Identifier, BinOp]:
        return self.expr()

# Demo and unit tests
if __name__ == "__main__":
    # Test lexer
    sample = "a + 3 * (b - 2)"
    lexer = Lexer(sample)
    tokens = lexer.tokenize()
    expected_types = [
        IDENTIFIER, PLUS, NUMBER, TIMES, LPAREN,
        IDENTIFIER, MINUS, NUMBER, RPAREN, EOF
    ]
    assert [t.type for t in tokens] == expected_types, "Lexer output mismatch"

    # Test parser
    parser = Parser(tokens)
    ast = parser.parse()
    # Simple structural checks
    assert isinstance(ast, BinOp), "Root should be BinOp"
    assert ast.op == '+', "Root operator should be '+'"
    assert isinstance(ast.left, Identifier) and ast.left.name == 'a', "Left operand should be identifier 'a'"
    assert isinstance(ast.right, BinOp), "Right operand should be BinOp"
    assert ast.right.op == '*', "Right operator should be '*'"
    assert isinstance(ast.right.left, Number) and ast.right.left.value == 3.0, "Left of '*' should be number 3"
    assert isinstance(ast.right.right, BinOp), "Right of '*' should be BinOp"
    assert ast.right.right.op == '-', "Inner operator should be '-'"
    assert isinstance(ast.right.right.left, Identifier) and ast.right.right.left.name == 'b', "Left of '-' should be 'b'"
    assert isinstance(ast.right.right.right, Number) and ast.right.right.right.value == 2.0, "Right of '-' should be 2"

    print("All tests passed.")
