"""
main.py
-------
Unit test suite, benchmarks, and entrypoint for the JIT simulation module.

All tests use only the standard library (unittest, time).
"""

from __future__ import annotations

import time
import unittest
from core import (
    OpCode,
    Instruction,
    BytecodeProgram,
    Interpreter,
    TieredJITCompiler,
    JITStats,
    JITMilestone,
    JIT_TIMELINE,
    get_milestones_in_range,
    get_milestone_by_year,
)


# ---------------------------------------------------------------------------
# Helper: build a simple program: push 3, push 4, add -> 7
# ---------------------------------------------------------------------------

def make_add_program(name: str = "add_3_4") -> BytecodeProgram:
    prog = BytecodeProgram(name=name)
    prog.append(OpCode.PUSH, 3)
    prog.append(OpCode.PUSH, 4)
    prog.append(OpCode.ADD)
    prog.append(OpCode.HALT)
    return prog


def make_noop_program(name: str = "noop") -> BytecodeProgram:
    """Program that pushes then pops (optimizable to empty)."""
    prog = BytecodeProgram(name=name)
    prog.append(OpCode.PUSH, 42)
    prog.append(OpCode.POP)
    prog.append(OpCode.HALT)
    return prog


def make_dup_program(name: str = "dup_test") -> BytecodeProgram:
    """Program with consecutive DUPs (optimizable)."""
    prog = BytecodeProgram(name=name)
    prog.append(OpCode.PUSH, 5)
    prog.append(OpCode.DUP)
    prog.append(OpCode.DUP)
    prog.append(OpCode.HALT)
    return prog


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestInterpreter(unittest.TestCase):
    """Tests for the stack-based interpreter."""

    def test_add_basic(self) -> None:
        prog = make_add_program()
        interp = Interpreter(prog)
        result = interp.run()
        self.assertEqual(result, 7)
        self.assertEqual(interp.executed_count, 4)  # PUSH, PUSH, ADD, HALT

    def test_sub_basic(self) -> None:
        prog = BytecodeProgram(name="sub")
        prog.append(OpCode.PUSH, 10)
        prog.append(OpCode.PUSH, 3)
        prog.append(OpCode.SUB)
        prog.append(OpCode.HALT)
        interp = Interpreter(prog)
        self.assertEqual(interp.run(), 7)

    def test_mul_basic(self) -> None:
        prog = BytecodeProgram(name="mul")
        prog.append(OpCode.PUSH, 6)
        prog.append(OpCode.PUSH, 7)
        prog.append(OpCode.MUL)
        prog.append(OpCode.HALT)
        interp = Interpreter(prog)
        self.assertEqual(interp.run(), 42)

    def test_dup_and_swap(self) -> None:
        prog = BytecodeProgram(name="dup_swap")
        prog.append(OpCode.PUSH, 1)
        prog.append(OpCode.PUSH, 2)
        prog.append(OpCode.SWAP)
        prog.append(OpCode.DUP)
        prog.append(OpCode.HALT)
        interp = Interpreter(prog)
        interp.run()
        # Stack after: [2, 1, 1] -> top is 1
        self.assertEqual(interp.stack[-1], 1)
        self.assertEqual(interp.stack, [2, 1, 1])

    def test_jump(self) -> None:
        prog = BytecodeProgram(name="jump")
        prog.append(OpCode.PUSH, 1)   # 0
        prog.append(OpCode.JUMP, 3)   # 1 -> jump to index 3
        prog.append(OpCode.PUSH, 99)  # 2 (skipped)
        prog.append(OpCode.HALT)      # 3
        interp = Interpreter(prog)
        result = interp.run()
        self.assertEqual(result, 1)

    def test_jump_if_zero(self) -> None:
        prog = BytecodeProgram(name="jz")
        prog.append(OpCode.PUSH, 0)   # 0
        prog.append(OpCode.JUMP_IF_ZERO, 3)  # 1 -> jump to 3 if 0
        prog.append(OpCode.PUSH, 99)  # 2 (skipped)
        prog.append(OpCode.HALT)      # 3
        interp = Interpreter(prog)
        result = interp.run()
        self.assertEqual(result, 0)

    def test_call_ret(self) -> None:
        prog = BytecodeProgram(name="call_ret")
        prog.append(OpCode.PUSH, 5)   # 0
        prog.append(OpCode.CALL, 3)   # 1 -> call at 3
        prog.append(OpCode.HALT)      # 2
        prog.append(OpCode.PUSH, 10)  # 3
        prog.append(OpCode.RET)       # 4
        interp = Interpreter(prog)
        result = interp.run()
        self.assertEqual(result, 10)

    def test_stack_underflow_raises(self) -> None:
        prog = BytecodeProgram(name="underflow")
        prog.append(OpCode.POP)
        prog.append(OpCode.HALT)
        interp = Interpreter(prog)
        with self.assertRaises(RuntimeError):
            interp.run()

    def test_stack_overflow_raises(self) -> None:
        prog = BytecodeProgram(name="overflow")
        for _ in range(Interpreter.MAX_STACK_DEPTH + 1):
            prog.append(OpCode.PUSH, 1)
        prog.append(OpCode.HALT)
        interp = Interpreter(prog)
        with self.assertRaises(RuntimeError):
            interp.run()


class TestTieredJITCompiler(unittest.TestCase):
    """Tests for the tiered JIT compiler simulation."""

    def test_interpretation_before_threshold(self) -> None:
        jit = TieredJITCompiler(threshold=3)
        prog = make_add_program()
        jit.register(prog)

        # First two calls: interpreted
        for _ in range(2):
            result = jit.execute("add_3_4")
            self.assertEqual(result, 7)

        stats = jit.get_stats("add_3_4")
        self.assertEqual(stats.interpreted_count, 2)
        self.assertEqual(stats.compiled_count, 0)
        self.assertEqual(stats.call_count, 2)

    def test_compilation_after_threshold(self) -> None:
        jit = TieredJITCompiler(threshold=3)
        prog = make_add_program()
        jit.register(prog)

        # Three calls: first two interpreted, third triggers compilation
        for _ in range(3):
            result = jit.execute("add_3_4")
            self.assertEqual(result, 7)

        stats = jit.get_stats("add_3_4")
        self.assertEqual(stats.interpreted_count, 2)
        self.assertEqual(stats.compiled_count, 1)
        self.assertEqual(stats.call_count, 3)
        # Optimized program should be smaller or equal
        self.assertLessEqual(stats.optimized_size, stats.original_size)

    def test_optimization_folds_constants(self) -> None:
        jit = TieredJITCompiler(threshold=1)
        prog = make_add_program()
        jit.register(prog)

        # Trigger compilation
        jit.execute("add_3_4")

        stats = jit.get_stats("add_3_4")
        # Original: PUSH 3, PUSH 4, ADD, HALT = 4 instructions
        # Optimized: PUSH 7, HALT = 2 instructions
        self.assertEqual(stats.original_size, 4)
        self.assertEqual(stats.optimized_size, 2)

    def test_optimization_removes_noop(self) -> None:
        jit = TieredJITCompiler(threshold=1)
        prog = make_noop_program()
        jit.register(prog)

        jit.execute("noop")

        stats = jit.get_stats("noop")
        # Original: PUSH 42, POP, HALT = 3
        # Optimized: HALT = 1
        self.assertEqual(stats.original_size, 3)
        self.assertEqual(stats.optimized_size, 1)

    def test_optimization_collapses_dups(self) -> None:
        jit = TieredJITCompiler(threshold=1)
        prog = make_dup_program()
        jit.register(prog)

        jit.execute("dup_test")

        stats = jit.get_stats("dup_test")
        # Original: PUSH 5, DUP, DUP, HALT = 4
        # Optimized: PUSH 5, DUP, HALT = 3
        self.assertEqual(stats.original_size, 4)
        self.assertEqual(stats.optimized_size, 3)

    def test_unregistered_program_raises(self) -> None:
        jit = TieredJITCompiler(threshold=3)
        with self.assertRaises(KeyError):
            jit.execute("nonexistent")

    def test_multiple_programs_independent(self) -> None:
        jit = TieredJITCompiler(threshold=2)
        prog1 = make_add_program("prog1")
        prog2 = make_noop_program("prog2")
        jit.register(prog1)
        jit.register(prog2)

        jit.execute("prog1")
        jit.execute("prog2")

        s1 = jit.get_stats("prog1")
        s2 = jit.get_stats("prog2")
        self.assertEqual(s1.call_count, 1)
        self.assertEqual(s2.call_count, 1)
        self.assertEqual(s1.interpreted_count, 1)
        self.assertEqual(s2.interpreted_count, 1)


class TestHistoricalTimeline(unittest.TestCase):
    """Tests for the JIT historical timeline data structure."""

    def test_timeline_not_empty(self) -> None:
        self.assertGreater(len(JIT_TIMELINE), 0)

    def test_timeline_sorted_by_year(self) -> None:
        years = [m.year for m in JIT_TIMELINE]
        self.assertEqual(years, sorted(years))

    def test_milestones_in_range(self) -> None:
        results = get_milestones_in_range(1990, 2000)
        self.assertGreater(len(results), 0)
        for m in results:
            self.assertGreaterEqual(m.year, 1990)
            self.assertLessEqual(m.year, 2000)

    def test_milestones_empty_range(self) -> None:
        results = get_milestones_in_range(1900, 1950)
        self.assertEqual(len(results), 0)

    def test_milestone_by_year_found(self) -> None:
        m = get_milestone_by_year(1997)
        self.assertIsNotNone(m)
        self.assertEqual(m.year, 1997)
        self.assertIn("HotSpot", m.title)

    def test_milestone_by_year_not_found(self) -> None:
        m = get_milestone_by_year(1950)
        self.assertIsNone(m)

    def test_all_milestones_have_valid_fields(self) -> None:
        for m in JIT_TIMELINE:
            self.assertIsInstance(m, JITMilestone)
            self.assertGreaterEqual(m.year, 1960)
            self.assertLessEqual(m.year, 2000)
            self.assertGreater(len(m.title), 0)
            self.assertGreater(len(m.description), 0)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_jit_vs_interpreter(iterations: int = 10000) -> Dict[str, float]:
    """
    Benchmark: compare total instruction count for interpreted vs JIT-compiled
    execution of a constant-folding program.

    Returns a dict with timing and instruction counts.
    """
    prog = make_add_program("bench")

    # Pure interpretation
    start = time.perf_counter()
    total_interp_instrs = 0
    for _ in range(iterations):
        interp = Interpreter(prog)
        interp.run()
        total_interp_instrs += interp.executed_count
    interp_time = time.perf_counter() - start

    # JIT with threshold=1 (compile immediately)
    jit = TieredJITCompiler(threshold=1)
    jit.register(prog)
    start = time.perf_counter()
    for _ in range(iterations):
        jit.execute("bench")
    jit_time = time.perf_counter() - start

    stats = jit.get_stats("bench")
    return {
        "iterations": iterations,
        "interpret_time_s": interp_time,
        "jit_time_s": jit_time,
        "total_interp_instructions": total_interp_instrs,
        "total_jit_instructions": stats.total_instructions_executed,
        "original_size": stats.original_size,
        "optimized_size": stats.optimized_size,
    }


# ---------------------------------------------------------------------------
# Demo / Entrypoint
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """Run a demonstration of the JIT simulation and historical timeline."""
    print("=" * 60)
    print("JIT Simulation Demo: 1960-2000 Historical Context")
    print("=" * 60)

    # 1. Demonstrate tiered compilation
    print("\n[1] Tiered JIT Compilation Demo")
    jit = TieredJITCompiler(threshold=3)
    prog = make_add_program("demo_add")
    jit.register(prog)

    for i in range(5):
        result = jit.execute("demo_add")
        stats = jit.get_stats("demo_add")
        tier = "JIT" if stats.compiled_count > 0 and i >= 2 else "Interp"
        print(f"  Call {i+1}: result={result}, tier={tier}, "
              f"interp={stats.interpreted_count}, compiled={stats.compiled_count}")

    stats = jit.get_stats("demo_add")
    print(f"  Original size: {stats.original_size}, "
          f"Optimized size: {stats.optimized_size}")

    # 2. Demonstrate optimization passes
    print("\n[2] Optimization Passes")
    jit2 = TieredJITCompiler(threshold=1)
    noop = make_noop_program("demo_noop")
    jit2.register(noop)
    jit2.execute("demo_noop")
    s2 = jit2.get_stats("demo_noop")
    print(f"  No-op program: {s2.original_size} -> {s2.optimized_size} instructions")

    dup = make_dup_program("demo_dup")
    jit2.register(dup)
    jit2.execute("demo_dup")
    s3 = jit2.get_stats("demo_dup")
    print(f"  Dup program:   {s3.original_size} -> {s3.optimized_size} instructions")

    # 3. Historical timeline
    print("\n[3] JIT Historical Timeline (1960-2000)")
    for m in JIT_TIMELINE:
        print(f"  {m.year}: {m.title}")
        print(f"         {m.description[:70]}...")

    # 4. Benchmark
    print("\n[4] Benchmark (10,000 iterations)")
    bench = benchmark_jit_vs_interpreter(10000)
    print(f"  Interpret time:  {bench['interpret_time_s']:.6f}s")
    print(f"  JIT time:        {bench['jit_time_s']:.6f}s")
    print(f"  Total interp instructions: {bench['total_interp_instructions']}")
    print(f"  Total JIT instructions:    {bench['total_jit_instructions']}")
    print(f"  Original size: {bench['original_size']}, "
          f"Optimized size: {bench['optimized_size']}")
    print("=" * 60)


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    print("\n")
    # Run demo
    run_demo()
