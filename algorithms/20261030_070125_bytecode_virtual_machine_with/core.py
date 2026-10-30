from __future__ import annotations
from enum import Enum, auto
from typing import Any, Callable, List, Tuple, Optional


class Opcode(Enum):
    PUSH_CONST = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()
    POP = auto()
    PRINT = auto()
    JUMP = auto()
    JUMP_IF_TRUE = auto()
    JUMP_IF_FALSE = auto()
    CALL = auto()
    RETURN = auto()
    HALT = auto()


Instruction = Tuple[Opcode, Any]  # operand may be None


class VMError(RuntimeError):
    """Base class for all VM runtime errors."""


class BytecodeVM:
    """
    A minimal stack‑based bytecode virtual machine.
    The VM is deterministic, pure‑Python and uses only the standard library.
    """

    def __init__(self, program: List[Instruction], globals_: Optional[dict] = None):
        self.program: List[Instruction] = program
        self.globals: dict = globals_ if globals_ is not None else {}
        self.stack: List[Any] = []
        self.call_stack: List[int] = []
        self.ip: int = 0
        self.output: List[Any] = []          # Captured values from PRINT
        self._dispatch: dict[Opcode, Callable[[Any], None]] = {
            Opcode.PUSH_CONST: self._op_push_const,
            Opcode.ADD: self._op_binary(lambda a, b: a + b),
            Opcode.SUB: self._op_binary(lambda a, b: a - b),
            Opcode.MUL: self._op_binary(lambda a, b: a * b),
            Opcode.DIV: self._op_binary(self._safe_div),
            Opcode.NEG: self._op_unary(lambda a: -a),
            Opcode.POP: self._op_pop,
            Opcode.PRINT: self._op_print,
            Opcode.JUMP: self._op_jump,
            Opcode.JUMP_IF_TRUE: self._op_jump_if(True),
            Opcode.JUMP_IF_FALSE: self._op_jump_if(False),
            Opcode.CALL: self._op_call,
            Opcode.RETURN: self._op_return,
            Opcode.HALT: self._op_halt,
        }
        self._halted: bool = False

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def run(self) -> Any:
        """Execute the loaded program until HALT or program end."""
        while not self._halted and 0 <= self.ip < len(self.program):
            self._step()
        # By convention the top of the stack is the result
        return self.stack[-1] if self.stack else None

    # --------------------------------------------------------------------- #
    # Core execution loop
    # --------------------------------------------------------------------- #
    def _step(self) -> None:
        instr = self.program[self.ip]
        opcode, operand = instr
        if opcode not in self._dispatch:
            raise VMError(f"Unknown opcode {opcode}")
        # Debug hook could be added here
        self._dispatch[opcode](operand)
        # Normal flow increments IP unless the opcode already changed it
        if not self._halted and opcode not in {
            Opcode.JUMP,
            Opcode.JUMP_IF_TRUE,
            Opcode.JUMP_IF_FALSE,
            Opcode.CALL,
            Opcode.RETURN,
        }:
            self.ip += 1

    # --------------------------------------------------------------------- #
    # Stack helpers
    # --------------------------------------------------------------------- #
    def _push(self, value: Any) -> None:
        self.stack.append(value)

    def _pop(self) -> Any:
        if not self.stack:
            raise VMError("Pop from empty stack")
        return self.stack.pop()

    # --------------------------------------------------------------------- #
    # Opcode implementations
    # --------------------------------------------------------------------- #
    def _op_push_const(self, operand: Any) -> None:
        self._push(operand)

    def _op_binary(self, func: Callable[[Any, Any], Any]) -> Callable[[Any], None]:
        def impl(_: Any) -> None:
            b = self._pop()
            a = self._pop()
            self._push(func(a, b))
        return impl

    def _op_unary(self, func: Callable[[Any], Any]) -> Callable[[Any], None]:
        def impl(_: Any) -> None:
            a = self._pop()
            self._push(func(a))
        return impl

    @staticmethod
    def _safe_div(a: Any, b: Any) -> Any:
        if b == 0:
            raise VMError("Division by zero")
        return a / b

    def _op_pop(self, _: Any) -> None:
        self._pop()

    def _op_print(self, _: Any) -> None:
        value = self._pop()
        self.output.append(value)

    def _op_jump(self, target: int) -> None:
        self._validate_target(target)
        self.ip = target

    def _op_jump_if(self, condition_is_true: bool) -> Callable[[Any], None]:
        def impl(target: int) -> None:
            self._validate_target(target)
            cond = self._pop()
            if bool(cond) is condition_is_true:
                self.ip = target
            else:
                self.ip += 1
        return impl

    def _op_call(self, target: int) -> None:
        self._validate_target(target)
        self.call_stack.append(self.ip + 1)
        self.ip = target

    def _op_return(self, _: Any) -> None:
        if not self.call_stack:
            raise VMError("Return with empty call stack")
        self.ip = self.call_stack.pop()

    def _op_halt(self, _: Any) -> None:
        self._halted = True

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #
    def _validate_target(self, target: int) -> None:
        if not (0 <= target < len(self.program)):
            raise VMError(f"Jump target {target} out of bounds")
