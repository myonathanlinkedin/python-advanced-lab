import unittest
from core import BytecodeVM, Opcode, VMError


class TestBytecodeVM(unittest.TestCase):
    def test_arithmetic(self):
        prog = [
            (Opcode.PUSH_CONST, 2),
            (Opcode.PUSH_CONST, 3),
            (Opcode.ADD, None),
            (Opcode.HALT, None),
        ]
        vm = BytecodeVM(prog)
        result = vm.run()
        self.assertEqual(result, 5)

    def test_unary_neg(self):
        prog = [
            (Opcode.PUSH_CONST, 4),
            (Opcode.NEG, None),
            (Opcode.HALT, None),
        ]
        vm = BytecodeVM(prog)
        self.assertEqual(vm.run(), -4)

    def test_division_and_error(self):
        prog = [
            (Opcode.PUSH_CONST, 10),
            (Opcode.PUSH_CONST, 2),
            (Opcode.DIV, None),
            (Opcode.HALT, None),
        ]
        vm = BytecodeVM(prog)
        self.assertEqual(vm.run(), 5.0)

        prog_err = [
            (Opcode.PUSH_CONST, 1),
            (Opcode.PUSH_CONST, 0),
            (Opcode.DIV, None),
            (Opcode.HALT, None),
        ]
        vm_err = BytecodeVM(prog_err)
        with self.assertRaises(VMError):
            vm_err.run()

    def test_control_flow(self):
        # Compute max(7, 3) using conditional jump
        prog = [
            (Opcode.PUSH_CONST, 7),          # 0
            (Opcode.PUSH_CONST, 3),          # 1
            (Opcode.SUB, None),              # 2 -> 4
            (Opcode.JUMP_IF_FALSE, 6),       # 3 if 7-3 <=0 jump to else
            (Opcode.PUSH_CONST, 7),          # 4 then branch result 7
            (Opcode.JUMP, 7),                # 5 skip else
            (Opcode.PUSH_CONST, 3),          # 6 else branch result 3
            (Opcode.HALT, None),             # 7
        ]
        vm = BytecodeVM(prog)
        self.assertEqual(vm.run(), 7)

    def test_function_call(self):
        # Simple function at address 5: adds two numbers and returns
        prog = [
            (Opcode.PUSH_CONST, 10),   # 0
            (Opcode.PUSH_CONST, 20),   # 1
            (Opcode.CALL, 5),          # 2 -> call add_two
            (Opcode.HALT, None),       # 3
            # padding
            (Opcode.POP, None),        # 4 (unused)
            # function add_two:
            (Opcode.PUSH_CONST, None), # 5 placeholder for arg1 (will be on stack)
            (Opcode.PUSH_CONST, None), # 6 placeholder for arg2
            (Opcode.ADD, None),        # 7
            (Opcode.RETURN, None),     # 8
        ]
        # To simulate passing arguments, we push them before CALL; CALL does not modify stack.
        vm = BytecodeVM(prog)
        self.assertEqual(vm.run(), 30)

    def test_print_capture(self):
        prog = [
            (Opcode.PUSH_CONST, "hello"),
            (Opcode.PRINT, None),
            (Opcode.PUSH_CONST, 42),
            (Opcode.PRINT, None),
            (Opcode.HALT, None),
        ]
        vm = BytecodeVM(prog)
        vm.run()
        self.assertEqual(vm.output, ["hello", 42])

    def test_invalid_opcode(self):
        prog = [(Opcode.HALT, None), ("INVALID", None)]  # type: ignore
        vm = BytecodeVM(prog)  # program contains an invalid opcode
        with self.assertRaises(VMError):
            vm.run()


if __name__ == "__main__":
    unittest.main()
