from __future__ import annotations

import threading
import random
import sys
from typing import List

from custom_types import SharedBuffer, Instruction
from engine import ChronosInterpreter

def _run_parallel_reads(buffer: SharedBuffer, offsets: List[int], results: List[int], idx: int) -> None:
    """
    Thread target that reads integers from the buffer without acquiring the buffer lock.
    This demonstrates zero‑copy, lock‑free reads via memoryview.
    """
    view = buffer.get_view()
    # each integer occupies 4 bytes; struct.unpack_from reads directly from the view
    for off in offsets:
        val = struct.unpack_from("<i", view, off)[0]
        results[idx].append(val)

def _run_parallel_writes(buffer: SharedBuffer, writes: List[tuple]) -> None:
    """
    Thread target that performs a series of writes using the buffer's thread‑safe API.
    """
    for offset, value in writes:
        buffer.write_int(offset, value)

def _basic_functionality_test() -> None:
    """
    Verify that the interpreter correctly executes a deterministic program.
    """
    buf = SharedBuffer(64)  # enough for 16 integers
    interpreter = ChronosInterpreter(buf)

    prog = [
        Instruction("SET", (0, 10)),      # mem[0] = 10
        Instruction("SET", (4, 20)),      # mem[4] = 20
        Instruction("ADD", (8, 0, 4)),    # mem[8] = mem[0] + mem[4] => 30
        Instruction("MUL", (12, 8, 0)),   # mem[12] = mem[8] * mem[0] => 300
        Instruction("GET", (12,)),        # retrieve 300
    ]

    results = interpreter.execute(prog)
    assert results == [300], f"Expected [300], got {results}"

def _concurrent_access_test() -> None:
    """
    Stress test: multiple writer threads update distinct slots while multiple reader
    threads concurrently read them. All reads must eventually see the values written
    by the writers (no torn reads).
    """
    buf = SharedBuffer(32)  # 8 integers
    writer_threads = []
    reader_threads = []
    num_writers = 4
    num_readers = 4

    # Prepare deterministic write sequences
    writes_per_thread = [
        [(i * 4, (i + 1) * 100 + t) for i in range(num_writers)]  # each thread writes to all slots
        for t in range(num_writers)
    ]

    # Launch writers
    for t in range(num_writers):
        th = threading.Thread(target=_run_parallel_writes, args=(buf, writes_per_thread[t]), daemon=True)
        writer_threads.append(th)
        th.start()

    # Prepare containers for reader results
    read_results: List[List[int]] = [[] for _ in range(num_readers)]
    offsets = [i * 4 for i in range(num_writers)]

    # Launch readers
    for r in range(num_readers):
        th = threading.Thread(target=_run_parallel_reads, args=(buf, offsets, read_results, r), daemon=True)
        reader_threads.append(th)
        th.start()

    # Wait for all threads to finish
    for th in writer_threads + reader_threads:
        th.join()

    # After all writes, each slot should contain the value written by the *last* writer thread
    expected = [(i + 1) * 100 + (num_writers - 1) for i in range(num_writers)]
    for i, off in enumerate(offsets):
        actual = buf.read_int(off)
        assert actual == expected[i], f"Slot {i} expected {expected[i]}, got {actual}"

    # Reader results must be a subset of the values that existed at some point.
    # Since writers may interleave, any observed value must belong to the set of possible writes.
    possible_values = {val for seq in writes_per_thread for _, val in seq}
    for r_idx, lst in enumerate(read_results):
        for val in lst:
            assert val in possible_values, f"Reader {r_idx} observed impossible value {val}"

def _edge_case_test() -> None:
    """
    Validate proper error handling for out‑of‑bounds accesses and division by zero.
    """
    buf = SharedBuffer(8)  # two integers
    interpreter = ChronosInterpreter(buf)

    # Out‑of‑bounds SET
    try:
        interpreter.execute([Instruction("SET", (8, 1))])  # offset 8 is exactly size, illegal
        assert False, "Expected IndexError for out‑of‑bounds SET"
    except IndexError:
        pass

    # Division by zero
    prog = [
        Instruction("SET", (0, 10)),
        Instruction("SET", (4, 0)),
        Instruction("DIV", (0, 0, 4)),
    ]
    try:
        interpreter.execute(prog)
        assert False, "Expected ZeroDivisionError for DIV by zero"
    except ZeroDivisionError:
        pass

def _demo() -> None:
    """
    Simple interactive demonstration executed when the module is run directly.
    """
    buf = SharedBuffer(16)  # four integers
    interpreter = ChronosInterpreter(buf)

    program = [
        Instruction("SET", (0, 7)),
        Instruction("SET", (4, 3)),
        Instruction("MUL", (8, 0, 4)),   # 7 * 3 = 21
        Instruction("SUB", (12, 8, 0)),  # 21 - 7 = 14
        Instruction("GET", (12,)),
    ]

    result = interpreter.execute(program)
    print("Chronos Interpreter Demo:")
    print(f"Final GET result = {result[0]} (expected 14)")

if __name__ == "__main__":
    # Run exhaustive unit‑tests
    _basic_functionality_test()
    _concurrent_access_test()
    _edge_case_test()
    # Show a short demo
    _demo()
    print("All tests passed successfully.")
