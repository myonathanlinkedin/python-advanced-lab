from __future__ import annotations

from typing import List, Dict, Callable
import struct

from custom_types import SharedBuffer, Instruction


class ChronosInterpreter:
    """
    A minimal, deterministic interpreter that operates directly on a SharedBuffer.
    All operations are zero‑copy: they read/write via memoryview or struct packing.
    Supported operations:
        - SET offset value          : store a signed 32‑bit integer
        - ADD dest src1 src2        : dest = src1 + src2
        - SUB dest src1 src2        : dest = src1 - src2
        - MUL dest src1 src2        : dest = src1 * src2
        - DIV dest src1 src2        : integer division, raises ZeroDivisionError if src2 == 0
        - GET offset                : returns the integer at offset (used by tests)
    Offsets are byte positions; each integer occupies 4 bytes.
    """

    def __init__(self, buffer: SharedBuffer):
        self.buffer = buffer
        self._dispatch: Dict[str, Callable[[Instruction], Any]] = {
            "SET": self._op_set,
            "ADD": self._op_add,
            "SUB": self._op_sub,
            "MUL": self._op_mul,
            "DIV": self._op_div,
            "GET": self._op_get,
        }

    def execute(self, program: List[Instruction]) -> List[int]:
        """
        Execute a list of instructions sequentially.
        Returns a list of results produced by GET instructions, preserving order.
        """
        results: List[int] = []
        for instr in program:
            if instr.op not in self._dispatch:
                raise ValueError(f"Unsupported operation: {instr.op}")
            outcome = self._dispatch[instr.op](instr)
            if instr.op == "GET":
                results.append(outcome)  # type: ignore
        return results

    # -----------------------------------------------------------------
    # Operation implementations
    # -----------------------------------------------------------------
    def _op_set(self, instr: Instruction) -> None:
        offset, value = instr.args
        self.buffer.write_int(offset, value)

    def _op_add(self, instr: Instruction) -> None:
        dest, src1, src2 = instr.args
        a = self.buffer.read_int(src1)
        b = self.buffer.read_int(src2)
        self.buffer.write_int(dest, a + b)

    def _op_sub(self, instr: Instruction) -> None:
        dest, src1, src2 = instr.args
        a = self.buffer.read_int(src1)
        b = self.buffer.read_int(src2)
        self.buffer.write_int(dest, a - b)

    def _op_mul(self, instr: Instruction) -> None:
        dest, src1, src2 = instr.args
        a = self.buffer.read_int(src1)
        b = self.buffer.read_int(src2)
        self.buffer.write_int(dest, a * b)

    def _op_div(self, instr: Instruction) -> None:
        dest, src1, src2 = instr.args
        a = self.buffer.read_int(src1)
        b = self.buffer.read_int(src2)
        if b == 0:
            raise ZeroDivisionError("division by zero in DIV instruction")
        self.buffer.write_int(dest, a // b)

    def _op_get(self, instr: Instruction) -> int:
        (offset,) = instr.args
        return self.buffer.read_int(offset)
