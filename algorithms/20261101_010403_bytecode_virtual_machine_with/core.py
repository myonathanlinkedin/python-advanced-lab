from __future__ import annotations

from enum import IntEnum, auto
from typing import Any, List, Tuple


class OpCode(IntEnum):
    """Supported bytecode operations."""
    PUSH = auto()   # operand: value to push
    POP = auto()    # operand: None
    ADD = auto()    # operand: None
    SUB = auto()    # operand: None
    MUL = auto()    # operand: None
    DIV = auto()    # operand: None (integer division)
    PRINT = auto()  # operand: None (captures top of stack)
    HALT = auto()   # operand: None


Instruction = Tuple[OpCode, Any]


class BytecodeVM:
    """
    A minimal stack‑based virtual machine for integer arithmetic.
    The VM is deterministic, pure‑Python and uses only the standard library.
    """

    __slots__ = ("stack", "pc", "program", "output", "_halted")

    def __init__(self) -> None:
        self.stack: List[int] = []
        self.pc: int = 0
        self.program: List[Instruction] = []
        self.output: List[str] = []
        self._halted: bool = False

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def load_program(self, program: List[Instruction]) -> None:
        """Replace the current program with *program* and reset the VM."""
        self.program = program
        self.reset()

    def reset(self) -> None:
        """Reset execution state without discarding the loaded program."""
        self.stack.clear()
        self.pc = 0
        self.output.clear()
        self._halted = False

    def run(self) -> List[str]:
        """
        Execute the loaded program until HALT or an error occurs.
        Returns the captured output produced by PRINT instructions.
        """
        while not self._halted:
            self._step()
        return self.output.copy()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _step(self) -> None:
        """Execute a single instruction; raise on malformed program."""
        if self.pc >= len(self.program):
            raise RuntimeError("Program counter out of bounds without HALT.")
        opcode, operand = self.program[self.pc]
        self.pc += 1

        if opcode is OpCode.PUSH:
            self._push(operand)
        elif opcode is OpCode.POP:
            self._pop()
        elif opcode is OpCode.ADD:
            self._binary_op(lambda a, b: a + b)
        elif opcode is OpCode.SUB:
            self._binary_op(lambda a, b: a - b)
        elif opcode is OpCode.MUL:
            self._binary_op(lambda a, b: a * b)
        elif opcode is OpCode.DIV:
            self._binary_op(self._safe_div)
        elif opcode is OpCode.PRINT:
            self._print()
        elif opcode is OpCode.HALT:
            self._halted = True
        else:
            raise RuntimeError(f"Unknown opcode: {opcode}")

    # --------------------------------------------------------------------- #
    # Stack manipulation primitives
    # --------------------------------------------------------------------- #
    def _push(self, value: Any) -> None:
        if not isinstance(value, int):
            raise TypeError("PUSH operand must be an integer.")
        self.stack.append(value)

    def _pop(self) -> int:
        if not self.stack:
            raise RuntimeError("POP from empty stack.")
        return self.stack.pop()

    def _binary_op(self, func) -> None:
        """Pop two values, apply *func*, push the result."""
        if len(self.stack) < 2:
            raise RuntimeError("Insufficient values on stack for binary operation.")
        b = self._pop()
        a = self._pop()
        result = func(a, b)
        self.stack.append(result)

    @staticmethod
    def _safe_div(a: int, b: int) -> int:
        if b == 0:
            raise RuntimeError("Division by zero.")
        return a // b

    def _print(self) -> None:
        if not self.stack:
            raise RuntimeError("PRINT from empty stack.")
        self.output.append(str(self.stack[-1]))

    # --------------------------------------------------------------------- #
    # Introspection helpers (useful for debugging / testing)
    # --------------------------------------------------------------------- #
    def peek(self) -> int:
        """Return the top of the stack without modifying it."""
        if not self.stack:
            raise RuntimeError("Peek from empty stack.")
        return self.stack[-1]

    def dump_state(self) -> dict:
        """Return a snapshot of the VM's internal state."""
        return {
            "pc": self.pc,
            "stack": self.stack.copy(),
            "output": self.output.copy(),
            "halted": self._halted,
        }
