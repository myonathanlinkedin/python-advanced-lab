import time
import unittest
from typing import List

from core import BytecodeVM, OpCode, Instruction


def make_program(*instrs: Instruction) -> List[Instruction]:
    """Utility to build a program list from variadic arguments."""
    return list(instrs)


class TestBytecodeVM(unittest.TestCase):
    def setUp(self) -> None:
        self.vm = BytecodeVM()

    def test_arithmetic(self) -> None:
        prog = make_program(
            (OpCode.PUSH, 10),
            (OpCode.PUSH, 5),
            (OpCode.ADD, None),      # 15
            (OpCode.PUSH, 2),
            (OpCode.MUL, None),      # 30
            (OpCode.PUSH, 4),
            (OpCode.SUB, None),      # 26
            (OpCode.PUSH, 2),
            (OpCode.DIV, None),      # 13
            (OpCode.HALT, None),
        )
        self.vm.load_program(prog)
        self.vm.run()
        self.assertEqual(self.vm.peek(), 13)

    def test_print_capture(self) -> None:
        prog = make_program(
            (OpCode.PUSH, 42),
            (OpCode.PRINT, None),
            (OpCode.PUSH, 7),
            (OpCode.PRINT, None),
            (OpCode.HALT, None),
        )
        self.vm.load_program(prog)
        out = self.vm.run()
        self.assertEqual(out, ["42", "7"])

    def test_pop_underflow(self) -> None:
        prog = make_program(
            (OpCode.POP, None),
            (OpCode.HALT, None),
        )
        self.vm.load_program(prog)
        with self.assertRaises(RuntimeError):
            self.vm.run()

    def test_division_by_zero(self) -> None:
        prog = make_program(
            (OpCode.PUSH, 5),
            (OpCode.PUSH, 0),
            (OpCode.DIV, None),
            (OpCode.HALT, None),
        )
        self.vm.load_program(prog)
        with self.assertRaises(RuntimeError):
            self.vm.run()

    def test_invalid_push_operand(self) -> None:
        prog = make_program(
            (OpCode.PUSH, "not-an-int"),
            (OpCode.HALT, None),
        )
        self.vm.load_program(prog)
        with self.assertRaises(TypeError):
            self.vm.run()

    def test_missing_halt(self) -> None:
        prog = make_program(
            (OpCode.PUSH, 1),
            (OpCode.PUSH, 2),
            (OpCode.ADD, None),
        )
        self.vm.load_program(prog)
        with self.assertRaises(RuntimeError):
            self.vm.run()

    def test_state_dump(self) -> None:
        prog = make_program(
            (OpCode.PUSH, 3),
            (OpCode.PUSH, 4),
            (OpCode.MUL, None),
            (OpCode.PRINT, None),
            (OpCode.HALT, None),
        )
        self.vm.load_program(prog)
        self.vm.run()
        state = self.vm.dump_state()
        self.assertTrue(state["halted"])
        self.assertEqual(state["stack"], [12])
        self.assertEqual(state["output"], ["12"])

def benchmark_simple_add(iterations: int = 100_000) -> float:
    """Run a tiny addition loop on the VM and return elapsed seconds."""
    prog = make_program(
        (OpCode.PUSH, 1),
        (OpCode.PUSH, 2),
        (OpCode.ADD, None),
        (OpCode.HALT, None),
    )
    vm = BytecodeVM()
    vm.load_program(prog)
    start = time.perf_counter()
    for _ in range(iterations):
        vm.reset()
        vm.run()
    return time.perf_counter() - start


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Simple benchmark demonstration
    elapsed = benchmark_simple_add(10_000)
    print(f"Benchmark: 10 000 executions in {elapsed:.4f} s")
