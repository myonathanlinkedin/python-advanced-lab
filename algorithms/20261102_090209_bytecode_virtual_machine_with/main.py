import unittest
from typing import List

from core import BytecodeVM, OpCode, Instruction


def assemble(*instrs: Instruction) -> List[Instruction]:
    """Utility to build a program list from variadic arguments."""
    return list(instrs)


class TestBytecodeVM(unittest.TestCase):
    def test_arithmetic(self):
        prog = assemble(
            (OpCode.PUSH, 10),
            (OpCode.PUSH, 5),
            (OpCode.ADD, None),
            (OpCode.PUSH, 2),
            (OpCode.MUL, None),
            (OpCode.PUSH, 3),
            (OpCode.SUB, None),
            (OpCode.PRINT, None),
            (OpCode.HALT, None),
        )
        vm = BytecodeVM(prog)
        vm.run()
        self.assertEqual(vm.stack, [23])          # ((10+5)*2)-3 = 23
        self.assertEqual(vm.output, [23])

    def test_control_flow(self):
        # Compute factorial of 4 using a loop
        prog = assemble(
            (OpCode.PUSH, 4),          # n
            (OpCode.PUSH, 1),          # result
            # loop_start (addr 2)
            (OpCode.DUP, None),        # duplicate n
            (OpCode.PUSH, 0),
            (OpCode.JZ, 12),           # if n == 0 jump to end
            (OpCode.SWAP, None),       # bring result under n
            (OpCode.MUL, None),        # result *= n
            (OpCode.PUSH, 1),
            (OpCode.SUB, None),        # n -= 1
            (OpCode.JUMP, 2),          # repeat
            # end (addr 12)
            (OpCode.POP, None),        # discard n (zero)
            (OpCode.PRINT, None),      # print result
            (OpCode.HALT, None),
        )
        vm = BytecodeVM(prog)
        vm.run()
        self.assertEqual(vm.output, [24])  # 4! = 24

    def test_stack_operations(self):
        prog = assemble(
            (OpCode.PUSH, 7),
            (OpCode.DUP, None),
            (OpCode.PUSH, 3),
            (OpCode.SWAP, None),
            (OpCode.SUB, None),   # 7 - 3 = 4
            (OpCode.PRINT, None),
            (OpCode.HALT, None),
        )
        vm = BytecodeVM(prog)
        vm.run()
        self.assertEqual(vm.stack, [4, 7])  # after SUB, stack: [4,7] (7 duplicated earlier)
        self.assertEqual(vm.output, [4])

    def test_error_handling(self):
        # Division by zero
        prog = assemble(
            (OpCode.PUSH, 10),
            (OpCode.PUSH, 0),
            (OpCode.DIV, None),
            (OpCode.HALT, None),
        )
        vm = BytecodeVM(prog)
        with self.assertRaises(RuntimeError) as ctx:
            vm.run()
        self.assertIn("Division by zero", str(ctx.exception))

        # Stack underflow
        prog2 = assemble(
            (OpCode.POP, None),
            (OpCode.HALT, None),
        )
        vm2 = BytecodeVM(prog2)
        with self.assertRaises(RuntimeError) as ctx2:
            vm2.run()
        self.assertIn("Stack underflow", str(ctx2.exception))

    def test_max_steps(self):
        # Infinite loop detection
        prog = assemble(
            (OpCode.JUMP, 0),   # jump to self
        )
        vm = BytecodeVM(prog)
        with self.assertRaises(RuntimeError) as ctx:
            vm.run(max_steps=1000)
        self.assertIn("Maximum step count exceeded", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
