from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Tuple


@dataclass(frozen=True)
class Label:
    """Represents a PTX label."""
    name: str


@dataclass(frozen=True)
class Instruction:
    """Represents a single PTX instruction (excluding labels)."""
    opcode: str                     # e.g. "add", "bra", "ret"
    type_suffix: str | None        # e.g. "s32" in "add.s32", None for no suffix
    operands: Tuple[str, ...]       # raw operand strings, e.g. ("%r1", "%r2", "%r3")
    raw: str                        # original source line (for debugging)


# A Kernel is a collection of instructions and labels that belong to a single entry point.
@dataclass
class Kernel:
    name: str
    body: List[Union[Label, Instruction]]   # ordered list preserving source order
