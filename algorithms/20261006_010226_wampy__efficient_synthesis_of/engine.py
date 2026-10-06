from __future__ import annotations
from typing import Generator, List, Optional, Tuple
from custom_types import Clause, Compound, Substitution, Var, Atom, Term


def occurs_check(var: Var, term: Term, subst: Substitution) -> bool:
    """Return True if var occurs in term after applying subst (prevent infinite terms)."""
    term = subst.apply(term)
    if var == term:
        return True
    if isinstance(term, Compound):
        return any(occurs_check(var, a, subst) for a in term.args)
    return False


def unify(t1: Term, t2: Term, subst: Substitution) -> Optional[Substitution]:
    """Attempt to unify t1 and t2 under subst. Return new substitution or None."""
    t1 = subst.apply(t1)
    t2 = subst.apply(t2)

    if isinstance(t1, Var):
        if t1 == t2:
            return subst
        if occurs_check(t1, t2, subst):
            return None
        new_subst = subst.copy()
        new_subst[t1] = t2
        return new_subst

    if isinstance(t2, Var):
        return unify(t2, t1, subst)

    if isinstance(t1, Atom) and isinstance(t2, Atom):
        if t1.name == t2.name:
            return subst
        return None

    if isinstance(t1, Compound) and isinstance(t2, Compound):
        if t1.functor != t2.functor or len(t1.args) != len(t2.args):
            return None
        for a1, a2 in zip(t1.args, t2.args):
            subst = unify(a1, a2, subst)
            if subst is None:
                return None
        return subst

    return None


class KnowledgeBase:
    """Container for Prolog clauses."""

    def __init__(self) -> None:
        self.clauses: List[Clause] = []

    def add(self, clause: Clause) -> None:
        self.clauses.append(clause)

    def __iter__(self):
        return iter(self.clauses)


class Solver:
    """Depth‑first Prolog resolver."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self._var_counter: List[int] = [0]  # mutable counter for fresh vars

    def query(self, goal: Compound) -> Generator[Substitution, None, None]:
        """Yield all substitutions that satisfy the goal."""
        initial_subst = Substitution()
        yield from self._resolve([goal], initial_subst)

    def _resolve(
        self, goals: List[Compound], subst: Substitution
    ) -> Generator[Substitution, None, None]:
        if not goals:
            yield subst
            return

        first, *rest = goals
        for clause in self.kb:
            renamed = clause.rename_vars(self._var_counter)
            new_subst = unify(first, renamed.head, subst)
            if new_subst is None:
                continue
            new_goals = list(renamed.body) + rest
            yield from self._resolve(new_goals, new_subst)
