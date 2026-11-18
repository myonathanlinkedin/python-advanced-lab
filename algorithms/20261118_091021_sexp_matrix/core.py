from __future__ import annotations
import operator
import re
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

# Simple S-expression representation
Sexp = Union[int, float, str, List["Sexp"]]

_OperatorFunc = Callable[[float, float], float]
_OPERATORS: Dict[str, _OperatorFunc] = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '^': operator.pow,
}


def _tokenize(s: str) -> List[str]:
    """Split a string into Lisp‑style tokens."""
    token_pattern = re.compile(r"""\s*(?:
        ([()] )            # parentheses
        |([^\s()]+)        # atoms
        )""", re.VERBOSE)
    tokens = []
    for paren, atom in token_pattern.findall(s):
        if paren:
            tokens.append(paren)
        elif atom:
            tokens.append(atom)
    return tokens


def _parse(tokens: List[str]) -> Tuple[Sexp, List[str]]:
    """Parse tokens into an S‑expression, returning the expression and the remaining tokens."""
    if not tokens:
        raise SyntaxError("Unexpected EOF while reading S‑expression")
    token = tokens.pop(0)
    if token == '(':
        lst: List[Sexp] = []
        while tokens and tokens[0] != ')':
            expr, tokens = _parse(tokens)
            lst.append(expr)
        if not tokens:
            raise SyntaxError("Missing closing ')'")
        tokens.pop(0)  # discard ')'
        return lst, tokens
    elif token == ')':
        raise SyntaxError("Unexpected ')'")
    else:
        # atom: try int, then float, else keep as string
        try:
            return int(token), tokens
        except ValueError:
            try:
                return float(token), tokens
            except ValueError:
                return token, tokens


def parse_sexp(s: str) -> Sexp:
    """Parse a Lisp‑style string into a nested Python list."""
    tokens = _tokenize(s)
    expr, remaining = _parse(tokens)
    if remaining:
        raise SyntaxError(f"Extra tokens after parsing: {remaining}")
    return expr


def sexp_to_str(sexp: Sexp) -> str:
    """Convert an S‑expression back to its string representation."""
    if isinstance(sexp, list):
        return '(' + ' '.join(sexp_to_str(e) for e in sexp) + ')'
    else:
        return str(sexp)


def evaluate_sexp(sexp: Sexp, env: Dict[str, Union[int, float]]) -> Union[int, float]:
    """Recursively evaluate an S‑expression given a variable environment."""
    if isinstance(sexp, (int, float)):
        return sexp
    if isinstance(sexp, str):
        if sexp in env:
            return env[sexp]
        raise NameError(f"Unbound variable '{sexp}'")
    if not isinstance(sexp, list) or not sexp:
        raise ValueError(f"Invalid S‑expression: {sexp}")

    op_token = sexp[0]
    if not isinstance(op_token, str) or op_token not in _OPERATORS:
        raise ValueError(f"Unsupported operator '{op_token}'")
    func = _OPERATORS[op_token]

    if len(sexp) != 3:
        raise ValueError(f"Operator '{op_token}' expects exactly two operands")
    left = evaluate_sexp(sexp[1], env)
    right = evaluate_sexp(sexp[2], env)
    return func(left, right)


class SexpMatrix:
    """A matrix whose entries are S‑expressions."""

    def __init__(self, rows: Sequence[Sequence[Sexp]]) -> None:
        if not rows:
            raise ValueError("Matrix must have at least one row")
        row_lengths = {len(r) for r in rows}
        if len(row_lengths) != 1:
            raise ValueError("All rows must have the same length")
        self._data: List[List[Sexp]] = [list(r) for r in rows]
        self._rows = len(self._data)
        self._cols = len(self._data[0])

    @property
    def shape(self) -> Tuple[int, int]:
        return self._rows, self._cols

    def __getitem__(self, idx: Tuple[int, int]) -> Sexp:
        i, j = idx
        return self._data[i][j]

    def __setitem__(self, idx: Tuple[int, int], value: Sexp) -> None:
        i, j = idx
        self._data[i][j] = value

    def __iter__(self):
        for row in self._data:
            yield row

    def __repr__(self) -> str:
        rows_str = ',\n  '.join(str([sexp_to_str(e) for e in row]) for row in self._data)
        return f"SexpMatrix([\n  {rows_str}\n])"

    def map(self, func: Callable[[Sexp], Sexp]) -> "SexpMatrix":
        """Return a new matrix with func applied to each entry."""
        new_rows = [[func(e) for e in row] for row in self._data]
        return SexpMatrix(new_rows)

    def __add__(self, other: "SexpMatrix") -> "SexpMatrix":
        if self.shape != other.shape:
            raise ValueError("Matrix dimensions must agree for addition")
        new_rows = []
        for r1, r2 in zip(self._data, other._data):
            new_row = [ ['+', a, b] for a, b in zip(r1, r2) ]
            new_rows.append(new_row)
        return SexpMatrix(new_rows)

    def __matmul__(self, other: "SexpMatrix") -> "SexpMatrix":
        """Matrix multiplication using symbolic '+' and '*' operators."""
        if self._cols != other._rows:
            raise ValueError("Inner dimensions must agree for multiplication")
        result_rows: List[List[Sexp]] = []
        for i in range(self._rows):
            result_row: List[Sexp] = []
            for j in range(other._cols):
                # compute sum_k (self[i,k] * other[k,j])
                terms: List[Sexp] = []
                for k in range(self._cols):
                    terms.append(['*', self._data[i][k], other._data[k][j]])
                # fold with '+'
                if not terms:
                    raise RuntimeError("Unexpected empty term list")
                expr = terms[0]
                for term in terms[1:]:
                    expr = ['+', expr, term]
                result_row.append(expr)
            result_rows.append(result_row)
        return SexpMatrix(result_rows)

    def evaluate(self, env: Dict[str, Union[int, float]]) -> List[List[Union[int, float]]]:
        """Evaluate every entry under the given environment, returning a plain numeric matrix."""
        return [[evaluate_sexp(e, env) for e in row] for row in self._data]
