from __future__ import annotations
from typing import Dict, List, Set, Tuple, Iterable


class Grammar:
    """
    Context‑free grammar in Chomsky Normal Form.

    Attributes
    ----------
    start: str
        The start non‑terminal.
    productions: Dict[str, List[Tuple[str, ...]]]
        Mapping from a non‑terminal to a list of right‑hand sides.
        Each RHS is a tuple of length 1 (terminal) or 2 (pair of non‑terminals).
    terminals: Set[str]
        Set of terminal symbols.
    nonterminals: Set[str]
        Set of non‑terminal symbols.
    """

    def __init__(self, start: str) -> None:
        self.start: str = start
        self.productions: Dict[str, List[Tuple[str, ...]]] = {}
        self.terminals: Set[str] = set()
        self.nonterminals: Set[str] = set()

    def add_production(self, lhs: str, rhs: Tuple[str, ...]) -> None:
        """Add a production lhs → rhs.  rhs must be length 1 or 2."""
        if len(rhs) not in (1, 2):
            raise ValueError("RHS must contain 1 or 2 symbols (CNF).")
        self.productions.setdefault(lhs, []).append(rhs)
        self.nonterminals.add(lhs)
        if len(rhs) == 1 and rhs[0].islower():
            self.terminals.add(rhs[0])
        else:
            for sym in rhs:
                if sym.isupper():
                    self.nonterminals.add(sym)
                else:
                    self.terminals.add(sym)

    @classmethod
    def from_rules(
        cls,
        start: str,
        rules: Iterable[Tuple[str, Tuple[str, ...]]],
    ) -> Grammar:
        """
        Build a Grammar from an iterable of (lhs, rhs) pairs.
        Example:
            Grammar.from_rules('S', [
                ('S', ('A', 'B')),
                ('A', ('a',)),
                ('B', ('b',)),
            ])
        """
        g = cls(start)
        for lhs, rhs in rules:
            g.add_production(lhs, rhs)
        return g

    def binary_rules(self) -> Dict[Tuple[str, str], Set[str]]:
        """Return mapping (B, C) → {A | A → B C}."""
        mapping: Dict[Tuple[str, str], Set[str]] = {}
        for lhs, rhss in self.productions.items():
            for rhs in rhss:
                if len(rhs) == 2:
                    key = (rhs[0], rhs[1])
                    mapping.setdefault(key, set()).add(lhs)
        return mapping

    def terminal_rules(self) -> Dict[str, Set[str]]:
        """Return mapping a → {A | A → a}."""
        mapping: Dict[str, Set[str]] = {}
        for lhs, rhss in self.productions.items():
            for rhs in rhss:
                if len(rhs) == 1 and rhs[0] in self.terminals:
                    mapping.setdefault(rhs[0], set()).add(lhs)
        return mapping


def cyk(grammar: Grammar, tokens: List[str]) -> bool:
    """
    CYK membership test.

    Parameters
    ----------
    grammar : Grammar
        Grammar in CNF.
    tokens : List[str]
        Input string as a list of terminal symbols.

    Returns
    -------
    bool
        True iff the start symbol derives the token sequence.
    """
    n = len(tokens)
    if n == 0:
        # CNF does not allow ε‑productions (except possibly for start,
        # which we deliberately ignore for simplicity).
        return False

    # Pre‑compute lookup tables for speed.
    term_rules = grammar.terminal_rules()
    bin_rules = grammar.binary_rules()

    # table[i][l] = set of non‑terminals deriving tokens[i : i+l+1]
    table: List[List[Set[str]]] = [
        [set() for _ in range(n)] for _ in range(n)
    ]

    # Length 1 substrings (terminals)
    for i, token in enumerate(tokens):
        table[i][0] = term_rules.get(token, set()).copy()

    # Build up longer substrings
    for length in range(2, n + 1):          # length of the span
        for i in range(n - length + 1):    # start of the span
            for split in range(1, length):  # split point
                left_set = table[i][split - 1]
                right_set = table[i + split][length - split - 1]
                if not left_set or not right_set:
                    continue
                for B in left_set:
                    for C in right_set:
                        for A in bin_rules.get((B, C), ()):
                            table[i][length - 1].add(A)

    # Acceptance condition
    return grammar.start in table[0][n - 1]


__all__ = ["Grammar", "cyk"]
