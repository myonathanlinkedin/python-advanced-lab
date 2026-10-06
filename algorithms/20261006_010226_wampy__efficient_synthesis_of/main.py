from __future__ import annotations
from typing import List, Tuple
from custom_types import Atom, Compound, Clause, Var, Substitution
from engine import KnowledgeBase, Solver, unify, occurs_check


def demo_program() -> KnowledgeBase:
    kb = KnowledgeBase()
    # Facts
    kb.add(Clause(Compound("parent", (Atom("john"), Atom("mary")))))
    kb.add(Clause(Compound("parent", (Atom("mary"), Atom("susan")))))
    # Rules
    # ancestor(X, Y) :- parent(X, Y).
    kb.add(
        Clause(
            Compound("ancestor", (Var("X"), Var("Y"))),
            (Compound("parent", (Var("X"), Var("Y"))),),
        )
    )
    # ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
    kb.add(
        Clause(
            Compound("ancestor", (Var("X"), Var("Y"))),
            (
                Compound("parent", (Var("X"), Var("Z"))),
                Compound("ancestor", (Var("Z"), Var("Y"))),
            ),
        )
    )
    return kb


def test_unify_basic() -> None:
    s = Substitution()
    X = Var("X")
    a = Atom("a")
    result = unify(X, a, s)
    assert result is not None
    assert result.mapping == {X: a}


def test_unify_compound() -> None:
    s = Substitution()
    X = Var("X")
    term1 = Compound("f", (X, Atom("b")))
    term2 = Compound("f", (Atom("a"), Var("Y")))
    result = unify(term1, term2, s)
    assert result is not None
    Y = Var("Y")
    assert result.mapping == {X: Atom("a"), Y: Atom("b")}


def test_occurs_check() -> None:
    s = Substitution()
    X = Var("X")
    term = Compound("f", (X,))
    # Trying to unify X with f(X) must fail
    assert unify(X, term, s) is None
    # Ensure occurs_check alone works
    assert occurs_check(X, term, s) is True


def test_ancestor_query() -> None:
    kb = demo_program()
    solver = Solver(kb)
    X = Var("X")
    query = Compound("ancestor", (Atom("john"), X))
    results = list(solver.query(query))
    # Extract concrete values for X
    answers = sorted({res.apply(X) for res in results})
    expected = sorted([Atom("mary"), Atom("susan")])
    assert answers == expected


def test_no_solution() -> None:
    kb = demo_program()
    solver = Solver(kb)
    query = Compound("ancestor", (Atom("susan"), Var("Who")))
    results = list(solver.query(query))
    assert len(results) == 0


def run_all_tests() -> None:
    test_unify_basic()
    test_unify_compound()
    test_occurs_check()
    test_ancestor_query()
    test_no_solution()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()
    # Demo interactive query
    kb = demo_program()
    solver = Solver(kb)
    Who = Var("Who")
    print("\nQuery: ancestor(john, Who)")
    for sol in solver.query(Compound("ancestor", (Atom("john"), Who))):
        print("  ", sol.apply(Who))
