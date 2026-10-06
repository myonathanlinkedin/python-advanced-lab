from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Iterable


class Term:
    """Base class for all Prolog terms."""

    def __repr__(self) -> str:
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError

    def vars(self) -> Iterable[Var]:
        """Yield all variables occurring in the term."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Var(Term):
    name: str

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def vars(self) -> Iterable[Var]:
        yield self


@dataclass(frozen=True, slots=True)
class Atom(Term):
    name: str

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Atom) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def vars(self) -> Iterable[Var]:
        return
        yield  # pragma: no cover


@dataclass(frozen=True, slots=True)
class Compound(Term):
    functor: str
    args: Tuple[Term, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        if not self.args:
            return self.functor
        args_str = ", ".join(repr(a) for a in self.args)
        return f"{self.functor}({args_str})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Compound)
            and self.functor == other.functor
            and self.args == other.args
        )

    def __hash__(self) -> int:
        return hash((self.functor, self.args))

    def vars(self) -> Iterable[Var]:
        for a in self.args:
            yield from a.vars()


@dataclass
class Substitution:
    mapping: Dict[Var, Term] = field(default_factory=dict)

    def __getitem__(self, var: Var) -> Term:
        return self.mapping[var]

    def __setitem__(self, var: Var, term: Term) -> None:
        self.mapping[var] = term

    def __contains__(self, var: Var) -> bool:
        return var in self.mapping

    def copy(self) -> Substitution:
        return Substitution(self.mapping.copy())

    def apply(self, term: Term) -> Term:
        """Recursively apply substitution to a term."""
        if isinstance(term, Var):
            if term in self.mapping:
                return self.apply(self.mapping[term])
            return term
        if isinstance(term, Atom):
            return term
        if isinstance(term, Compound):
            new_args = tuple(self.apply(a) for a in term.args)
            if new_args == term.args:
                return term
            return Compound(term.functor, new_args)
        raise TypeError(f"Unsupported term type: {type(term)}")

    def __repr__(self) -> str:
        items = ", ".join(f"{v}/{t}" for v, t in self.mapping.items())
        return f"{{{items}}}"


@dataclass(frozen=True, slots=True)
class Clause:
    """A Prolog clause: head :- body."""
    head: Compound
    body: Tuple[Compound, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        if not self.body:
            return f"{self.head}."
        body_str = ", ".join(repr(b) for b in self.body)
        return f"{self.head} :- {body_str}."

    def rename_vars(self, counter: List[int]) -> Clause:
        """Return a copy of the clause with fresh variables."""
        var_map: Dict[Var, Var] = {}

        def fresh(v: Var) -> Var:
            if v not in var_map:
                counter[0] += 1
                var_map[v] = Var(f"{v.name}_{counter[0]}")
            return var_map[v]

        def rename(term: Term) -> Term:
            if isinstance(term, Var):
                return fresh(term)
            if isinstance(term, Atom):
                return term
            if isinstance(term, Compound):
                return Compound(term.functor, tuple(rename(a) for a in term.args))
            raise TypeError

        new_head = rename(self.head)
        new_body = tuple(rename(b) for b in self.body)
        return Clause(new_head, new_body)
