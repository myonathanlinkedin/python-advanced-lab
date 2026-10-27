from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Set, List, FrozenSet


Symbol = str  # Alias for readability; both terminals and non‑terminals are strings


@dataclass(frozen=True, slots=True)
class Production:
    """
    A production rule of a context‑free grammar in Chomsky Normal Form.

    * ``lhs`` – left‑hand side non‑terminal.
    * ``rhs`` – right‑hand side tuple:
        - length 1 for a terminal production (A → a)
        - length 2 for a binary non‑terminal production (A → B C)
        - length 0 for the ε‑production allowed only for the start symbol.
    """
    lhs: Symbol
    rhs: Tuple[Symbol, ...]  # 0, 1 or 2 symbols


@dataclass
class Grammar:
    """
    A context‑free grammar in Chomsky Normal Form.

    Attributes
    ----------
    terminals: set of terminal symbols.
    nonterminals: set of non‑terminal symbols.
    productions: set of Production objects.
    start: the start non‑terminal.
    """
    terminals: Set[Symbol]
    nonterminals: Set[Symbol]
    productions: Set[Production]
    start: Symbol

    def __post_init__(self) -> None:
        # Basic sanity checks – raise early if the grammar is malformed.
        if self.start not in self.nonterminals:
            raise ValueError(f"Start symbol {self.start!r} must be a non‑terminal.")
        for prod in self.productions:
            if prod.lhs not in self.nonterminals:
                raise ValueError(f"LHS {prod.lhs!r} of production {prod} is not a non‑terminal.")
            if len(prod.rhs) == 0:
                # ε‑productions are allowed only for the start symbol.
                if prod.lhs != self.start:
                    raise ValueError("Only the start symbol may produce ε in CNF.")
            elif len(prod.rhs) == 1:
                if prod.rhs[0] not in self.terminals:
                    raise ValueError(f"Terminal production {prod} uses non‑terminal {prod.rhs[0]!r}.")
            elif len(prod.rhs) == 2:
                if not all(sym in self.nonterminals for sym in prod.rhs):
                    raise ValueError(f"Binary production {prod} must have non‑terminals on RHS.")
            else:
                raise ValueError(f"Production {prod} is not in CNF (RHS length {len(prod.rhs)}).")
