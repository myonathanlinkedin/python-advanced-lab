import enum
from typing import Any, List, Tuple, Optional


class OpCode(enum.IntEnum):
    """Supported bytecode operations."""
    NOP = 0          # No operation
    PUSH = 1         # Push immediate value onto stack
    POP = 2          # Pop top of stack (discard)
    DUP = 3          # Duplicate top of stack
    SWAP = 4         # Swap top two stack items
    ADD = 5          # a + b
    SUB = 6          # a - b
    MUL = 7          # a * b
    DIV = 8          # a // b (integer division)
    MOD = 9          # a % b
    NEG = 10         # -a
    PRINT = 11       # Print top of stack (kept)
    JUMP = 12        # Unconditional jump to address
    JZ = 13          # Jump if top == 0 (pop)
    JNZ = 14         # Jump if top != 0 (pop)
    HALT = 15        # Stop execution


Instruction = Tuple[OpCode, Optional[Any]]  # operand may be None


class BytecodeVM:
    """
    Simple stack‑based virtual machine executing a list of (opcode, operand) tuples.
    """

    def __init__(self, code: List[Instruction]) -> None:
        self.code: List[Instruction] = code
        self.ip: int = 0                     # instruction pointer
        self.stack: List[int] = []           # operand stack (int only)
        self.halted: bool = False
        self.output: List[int] = []          # values printed via PRINT

    # --------------------------------------------------------------------- #
    # Core execution primitives
    # --------------------------------------------------------------------- #
    def fetch(self) -> Instruction:
        if self.ip < 0 or self.ip >= len(self.code):
            raise RuntimeError(f"Instruction pointer out of bounds: {self.ip}")
        instr = self.code[self.ip]
        self.ip += 1
        return instr

    def push(self, value: int) -> None:
        self.stack.append(value)

    def pop(self) -> int:
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack.pop()

    # --------------------------------------------------------------------- #
    # Instruction implementations
    # --------------------------------------------------------------------- #
    def exec_nop(self, _: Any) -> None:
        pass

    def exec_push(self, operand: Any) -> None:
        if not isinstance(operand, int):
            raise RuntimeError(f"PUSH expects int operand, got {operand!r}")
        self.push(operand)

    def exec_pop(self, _: Any) -> None:
        self.pop()

    def exec_dup(self, _: Any) -> None:
        if not self.stack:
            raise RuntimeError("DUP on empty stack")
        self.push(self.stack[-1])

    def exec_swap(self, _: Any) -> None:
        if len(self.stack) < 2:
            raise RuntimeError("SWAP needs at least two stack items")
        self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]

    def binary_op(self, func) -> None:
        b = self.pop()
        a = self.pop()
        self.push(func(a, b))

    def exec_add(self, _: Any) -> None:
        self.binary_op(lambda a, b: a + b)

    def exec_sub(self, _: Any) -> None:
        self.binary_op(lambda a, b: a - b)

    def exec_mul(self, _: Any) -> None:
        self.binary_op(lambda a, b: a * b)

    def exec_div(self, _: Any) -> None:
        def safe_div(a: int, b: int) -> int:
            if b == 0:
                raise RuntimeError("Division by zero")
            return a // b
        self.binary_op(safe_div)

    def exec_mod(self, _: Any) -> None:
        def safe_mod(a: int, b: int) -> int:
            if b == 0:
                raise RuntimeError("Modulo by zero")
            return a % b
        self.binary_op(safe_mod)

    def exec_neg(self, _: Any) -> None:
        val = self.pop()
        self.push(-val)

    def exec_print(self, _: Any) -> None:
        if not self.stack:
            raise RuntimeError("PRINT on empty stack")
        val = self.stack[-1]
        self.output.append(val)

    def exec_jump(self, operand: Any) -> None:
        if not isinstance(operand, int):
            raise RuntimeError("JUMP operand must be int address")
        self.ip = operand

    def exec_jz(self, operand: Any) -> None:
        if not isinstance(operand, int):
            raise RuntimeError("JZ operand must be int address")
        val = self.pop()
        if val == 0:
            self.ip = operand

    def exec_jnz(self, operand: Any) -> None:
        if not isinstance(operand, int):
            raise RuntimeError("JNZ operand must be int address")
        val = self.pop()
        if val != 0:
            self.ip = operand

    def exec_halt(self, _: Any) -> None:
        self.halted = True

    # --------------------------------------------------------------------- #
    # Dispatch table
    # --------------------------------------------------------------------- #
    _dispatch = {
        OpCode.NOP: exec_nop,
        OpCode.PUSH: exec_push,
        OpCode.POP: exec_pop,
        OpCode.DUP: exec_dup,
        OpCode.SWAP: exec_swap,
        OpCode.ADD: exec_add,
        OpCode.SUB: exec_sub,
        OpCode.MUL: exec_mul,
        OpCode.DIV: exec_div,
        OpCode.MOD: exec_mod,
        OpCode.NEG: exec_neg,
        OpCode.PRINT: exec_print,
        OpCode.JUMP: exec_jump,
        OpCode.JZ: exec_jz,
        OpCode.JNZ: exec_jnz,
        OpCode.HALT: exec_halt,
    }

    # --------------------------------------------------------------------- #
    # Execution loop
    # --------------------------------------------------------------------- #
    def step(self) -> None:
        """Execute a single instruction."""
        if self.halted:
            return
        opcode, operand = self.fetch()
        handler = self._dispatch.get(opcode)
        if handler is None:
            raise RuntimeError(f"Invalid opcode: {opcode}")
        # Bind handler to self and invoke
        handler(self, operand)

    def run(self, max_steps: Optional[int] = None) -> None:
        """Run until HALT or optional step limit."""
        steps = 0
        while not self.halted:
            self.step()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                raise RuntimeError("Maximum step count exceeded")
        # After halt, ensure ip points to HALT (optional sanity)
        return
