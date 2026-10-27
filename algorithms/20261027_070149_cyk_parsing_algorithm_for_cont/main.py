from __future__ import annotations

from typing import List

from custom_types import Grammar, Production, Symbol
from engine import cyk


def build_a_n_b_n_grammar() -> Grammar:
    """
    Constructs a CNF grammar for the language { aⁿ bⁿ | n ≥ 0 }.
    The grammar includes an ε‑production for the empty string.

    Non‑terminals: {S, X, A, B}
    Terminals:    {a, b}
    Productions:
        S → A X
        S → ε
        X → S B
        A → a
        B → b
    """
    terminals = {"a", "b"}
    nonterminals = {"S", "X", "A", "B"}
    productions = {
        Production("S", ("A", "X")),
        Production("S", ()),               # ε‑production
        Production("X", ("S", "B")),
        Production("A", ("a",)),
        Production("B", ("b",)),
    }
    return Grammar(terminals, nonterminals, productions, start="S")


def test_cyk_membership() -> None:
    grammar = build_a_n_b_n_grammar()

    # Positive examples (should be accepted)
    positives: List[List[Symbol]] = [
        [],                     # ε
        ["a", "b"],
        ["a", "a", "b", "b"],
        ["a", "a", "a", "b", "b", "b"],
    ]

    for tokens in positives:
        assert cyk(grammar, tokens), f"CYK incorrectly rejected {tokens!r}"

    # Negative examples (should be rejected)
    negatives: List[List[Symbol]] = [
        ["a"],                  # unmatched a
        ["b"],                  # unmatched b
        ["a", "b", "b"],        # extra b
        ["a", "a", "b"],        # missing b
        ["b", "a"],             # wrong order
        ["a", "b", "a", "b"],   # interleaved
        ["c"],                  # symbol not in terminals
    ]

    for tokens in negatives:
        assert not cyk(grammar, tokens), f"CYK incorrectly accepted {tokens!r}"


def test_edge_cases() -> None:
    # Grammar with only an ε‑production.
    epsilon_grammar = Grammar(
        terminals=set(),
        nonterminals={"S"},
        productions={Production("S", ())},
        start="S",
    )
    assert cyk(epsilon_grammar, []), "Empty string should be accepted by ε‑grammar."
    assert not cyk(epsilon_grammar, ["a"]), "Non‑empty string must be rejected by ε‑grammar."

    # Grammar that cannot derive any terminal (no productions for terminals).
    dead_grammar = Grammar(
        terminals={"a"},
        nonterminals={"S"},
        productions={Production("S", ("S", "S"))},  # non‑productive binary rule
        start="S",
    )
    assert not cyk(dead_grammar, ["a"]), "String should be rejected when no terminal productions exist."


if __name__ == "__main__":
    test_cyk_membership()
    test_edge_cases()
    # Demo: parse a user‑provided string.
    demo_grammar = build_a_n_b_n_grammar()
    sample = ["a", "a", "a", "b", "b", "b"]
    result = cyk(demo_grammar, sample)
    print(f"The string {''.join(sample)} is {'accepted' if result else 'rejected'} by the grammar.")
