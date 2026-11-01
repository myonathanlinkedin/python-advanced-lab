"""
core.py
-------
A pure, standard-library-only reference implementation of a Just-In-Time (JIT)
compiler *simulation* that models the historical evolution of JIT techniques
from 1960 to 2000.

The module provides:
  1. A minimal bytecode IR (Instruction, BytecodeProgram).
  2. A stack-based interpreter (baseline execution).
  3. A tiered JIT compiler that:
       - profiles hot methods (call-count threshold),
       - performs a simple peephole / constant-folding optimization pass,
       - emits an "optimized" bytecode variant,
       - tracks per-tier execution statistics.
  4. A historical timeline data structure mapping JIT milestones to years.

All logic is deterministic, side-effect-free (except internal counters), and
uses only the Python standard library.

Time complexity:
  - Interpretation: O(n) per execution where n = number of instructions.
  - JIT compilation (optimization pass): O(n) single pass.
  - Overall program execution: O(total_instructions * tiers).
Space complexity: O(n) for bytecode storage; O(1) auxiliary per execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Bytecode IR
# ---------------------------------------------------------------------------

class OpCode(Enum):
    """Minimal opcode set for the simulated bytecode VM."""
    PUSH = auto()      # push immediate operand onto stack
    POP = auto()       # pop top of stack (discard)
    ADD = auto()       # pop two, push sum
    SUB = auto()       # pop two, push difference
    MUL = auto()       # pop two, push product
    DUP = auto()       # duplicate top of stack
    SWAP = auto()      # swap top two stack elements
    JUMP = auto()      # unconditional jump to operand index
    JUMP_IF_ZERO = auto()  # jump to operand if top of stack is zero
    CALL = auto()      # call subroutine at operand index
    RET = auto()       # return from subroutine
    HALT = auto()      # stop execution


@dataclass(frozen=True)
class Instruction:
    """A single bytecode instruction."""
    op: OpCode
    operand: int = 0  # immediate value or target index


@dataclass
class BytecodeProgram:
    """A named collection of bytecode instructions."""
    name: str
    instructions: List[Instruction] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.instructions)

    def append(self, op: OpCode, operand: int = 0) -> None:
        self.instructions.append(Instruction(op, operand))


# ---------------------------------------------------------------------------
# 2. Stack-based Interpreter
# ---------------------------------------------------------------------------

class Interpreter:
    """
    Executes a BytecodeProgram on a virtual stack machine.

    Invariant: stack depth never exceeds 256; program counter stays in bounds.
    Time: O(n) per full execution.
    """

    MAX_STACK_DEPTH = 256

    def __init__(self, program: BytecodeProgram) -> None:
        self.program = program
        self.stack: List[int] = []
        self.pc: int = 0
        self.call_stack: List[int] = []
        self.executed_count: int = 0

    def _push(self, value: int) -> None:
        if len(self.stack) >= self.MAX_STACK_DEPTH:
            raise RuntimeError("Stack overflow")
        self.stack.append(value)

    def _pop(self) -> int:
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack.pop()

    def run(self) -> int:
        """Execute until HALT or RET at top level. Returns final stack top."""
        while True:
            if self.pc >= len(self.program.instructions):
                break
            instr = self.program.instructions[self.pc]
            self.executed_count += 1
            op = instr.op

            if op == OpCode.PUSH:
                self._push(instr.operand)
                self.pc += 1
            elif op == OpCode.POP:
                self._pop()
                self.pc += 1
            elif op == OpCode.ADD:
                b = self._pop()
                a = self._pop()
                self._push(a + b)
                self.pc += 1
            elif op == OpCode.SUB:
                b = self._pop()
                a = self._pop()
                self._push(a - b)
                self.pc += 1
            elif op == OpCode.MUL:
                b = self._pop()
                a = self._pop()
                self._push(a * b)
                self.pc += 1
            elif op == OpCode.DUP:
                if not self.stack:
                    raise RuntimeError("Stack underflow on DUP")
                self._push(self.stack[-1])
                self.pc += 1
            elif op == OpCode.SWAP:
                if len(self.stack) < 2:
                    raise RuntimeError("Stack underflow on SWAP")
                self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
                self.pc += 1
            elif op == OpCode.JUMP:
                self.pc = instr.operand
            elif op == OpCode.JUMP_IF_ZERO:
                val = self._pop()
                if val == 0:
                    self.pc = instr.operand
                else:
                    self.pc += 1
            elif op == OpCode.CALL:
                self.call_stack.append(self.pc + 1)
                self.pc = instr.operand
            elif op == OpCode.RET:
                if not self.call_stack:
                    break
                self.pc = self.call_stack.pop()
            elif op == OpCode.HALT:
                break
            else:
                raise ValueError(f"Unknown opcode: {op}")

        return self.stack[-1] if self.stack else 0


# ---------------------------------------------------------------------------
# 3. Tiered JIT Compiler (simulation)
# ---------------------------------------------------------------------------

@dataclass
class JITStats:
    """Per-method JIT statistics."""
    name: str
    call_count: int = 0
    interpreted_count: int = 0
    compiled_count: int = 0
    original_size: int = 0
    optimized_size: int = 0
    total_instructions_executed: int = 0


class TieredJITCompiler:
    """
    Simulates a tiered JIT compiler with:
      - Tier 0: interpretation (baseline)
      - Tier 1: optimized bytecode (peephole + constant folding)

    Compilation is triggered when call_count >= threshold.
    The optimization pass is a single O(n) scan that:
      - Folds PUSH a; PUSH b; ADD  ->  PUSH (a+b)
      - Removes POP after PUSH (no-op pair)
      - Collapses consecutive DUPs
    """

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold
        self._programs: Dict[str, BytecodeProgram] = {}
        self._optimized: Dict[str, BytecodeProgram] = {}
        self._stats: Dict[str, JITStats] = {}

    def register(self, program: BytecodeProgram) -> None:
        """Register a program for JIT management."""
        self._programs[program.name] = program
        self._stats[program.name] = JITStats(
            name=program.name,
            original_size=len(program.instructions),
        )

    def _optimize(self, program: BytecodeProgram) -> BytecodeProgram:
        """
        Single-pass peephole optimization.
        Time: O(n). Space: O(n) for output list.
        """
        out: List[Instruction] = []
        i = 0
        n = len(program.instructions)
        while i < n:
            cur = program.instructions[i]
            # Pattern: PUSH a; PUSH b; ADD  ->  PUSH (a+b)
            if (
                cur.op == OpCode.PUSH
                and i + 2 < n
                and program.instructions[i + 1].op == OpCode.PUSH
                and program.instructions[i + 2].op == OpCode.ADD
            ):
                a = cur.operand
                b = program.instructions[i + 1].operand
                out.append(Instruction(OpCode.PUSH, a + b))
                i += 3
                continue
            # Pattern: PUSH a; POP  ->  (remove both)
            if (
                cur.op == OpCode.PUSH
                and i + 1 < n
                and program.instructions[i + 1].op == OpCode.POP
            ):
                i += 2
                continue
            # Pattern: consecutive DUPs -> single DUP
            if cur.op == OpCode.DUP and out and out[-1].op == OpCode.DUP:
                i += 1
                continue
            out.append(cur)
            i += 1

        return BytecodeProgram(name=program.name + "_opt", instructions=out)

    def execute(self, name: str) -> int:
        """
        Execute a registered program, applying JIT tiering.
        Returns the final stack value.
        """
        if name not in self._programs:
            raise KeyError(f"Program '{name}' not registered")

        stats = self._stats[name]
        stats.call_count += 1

        # Decide tier
        if stats.call_count >= self.threshold and name in self._optimized:
            # Tier 1: use optimized bytecode
            interp = Interpreter(self._optimized[name])
            result = interp.run()
            stats.compiled_count += 1
            stats.total_instructions_executed += interp.executed_count
        else:
            # Tier 0: interpret original
            interp = Interpreter(self._programs[name])
            result = interp.run()
            stats.interpreted_count += 1
            stats.total_instructions_executed += interp.executed_count

            # Trigger compilation after threshold reached
            if stats.call_count == self.threshold:
                self._optimized[name] = self._optimize(self._programs[name])
                stats.optimized_size = len(self._optimized[name].instructions)

        return result

    def get_stats(self, name: str) -> JITStats:
        return self._stats[name]

    def get_all_stats(self) -> Dict[str, JITStats]:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# 4. Historical Timeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JITMilestone:
    """A historical JIT milestone."""
    year: int
    title: str
    description: str


JIT_TIMELINE: List[JITMilestone] = [
    JITMilestone(1960, "Early Interpreters",
                 "Lisp and early ALGOL systems used tree-walking interpreters; "
                 "no compilation, pure interpretation."),
    JITMilestone(1972, "Smalltalk-72 Bytecode VM",
                 "Dan Ingalls' Smalltalk-72 introduced a bytecode-based VM with "
                 "a stack machine, laying groundwork for later JIT designs."),
    JITMilestone(1983, "Smalltalk-80 / Squeak Precursors",
                 "Bytecode dispatch and method caching began appearing in "
                 "Smalltalk implementations; early 'hot method' detection."),
    JITMilestone(1990, "HotSpot Prototype (Sun)",
                 "Sun's HotSpot project began exploring adaptive compilation "
                 "with profiling-driven tiered compilation."),
    JITMilestone(1995, "JavaScript Engines (Netscape)",
                 "Netscape Navigator shipped a bytecode interpreter for JS; "
                 "later versions added simple JIT compilation for hot loops."),
    JITMilestone(1997, "HotSpot C1/C2 Architecture",
                 "HotSpot formalized C1 (client) and C2 (server) compiler tiers, "
                 "with deoptimization and on-stack replacement."),
    JITMilestone(1998, "JIT in .NET CLR (Early Design)",
                 "Microsoft's CLR design documents described a JIT compiler "
                 "that translates IL to native code at runtime."),
    JITMilestone(2000, "JIT Maturity (Java 1.3 / .NET Beta)",
                 "Java 1.3 shipped with a production JIT; .NET beta confirmed "
                 "JIT as a core runtime component. JIT became mainstream."),
]


def get_milestones_in_range(start_year: int, end_year: int) -> List[JITMilestone]:
    """Return milestones whose year falls in [start_year, end_year]."""
    return [m for m in JIT_TIMELINE if start_year <= m.year <= end_year]


def get_milestone_by_year(year: int) -> Optional[JITMilestone]:
    """Return the milestone for an exact year, or None."""
    for m in JIT_TIMELINE:
        if m.year == year:
            return m
    return None
