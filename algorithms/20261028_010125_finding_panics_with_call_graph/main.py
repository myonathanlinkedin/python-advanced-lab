"""
main.py
---------

Unit tests and demonstration for the CallGraph analyzer defined in core.py.
All tests are self‑contained and use only the Python standard library.
"""

import unittest
from typing import Set

from core import CallGraph


class TestCallGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.cg = CallGraph()

    def test_empty_graph(self) -> None:
        self.assertEqual(self.cg.find_panic_functions(), set())

    def test_single_panic_function(self) -> None:
        self.cg.add_function("foo", panics=True)
        self.assertEqual(self.cg.find_panic_functions(), {"foo"})

    def test_simple_chain(self) -> None:
        # a -> b -> c, where c panics.
        self.cg.add_call("a", "b")
        self.cg.add_call("b", "c")
        self.cg.add_function("c", panics=True)
        self.assertEqual(self.cg.find_panic_functions(), {"a", "b", "c"})

    def test_branching_calls(self) -> None:
        # a calls b and c; b calls d (panic); c calls e (no panic)
        self.cg.add_call("a", "b")
        self.cg.add_call("a", "c")
        self.cg.add_call("b", "d")
        self.cg.add_call("c", "e")
        self.cg.add_function("d", panics=True)
        self.cg.add_function("e", panics=False)
        self.assertEqual(self.cg.find_panic_functions(), {"a", "b", "d"})

    def test_multiple_panic_sources(self) -> None:
        # x -> y, y -> z (panic); u -> v (panic)
        self.cg.add_call("x", "y")
        self.cg.add_call("y", "z")
        self.cg.add_call("u", "v")
        self.cg.add_function("z", panics=True)
        self.cg.add_function("v", panics=True)
        self.assertEqual(self.cg.find_panic_functions(), {"x", "y", "z", "u", "v"})

    def test_cycle_including_panic(self) -> None:
        # a -> b -> c -> a (cycle), c panics.
        self.cg.add_call("a", "b")
        self.cg.add_call("b", "c")
        self.cg.add_call("c", "a")
        self.cg.add_function("c", panics=True)
        self.assertEqual(self.cg.find_panic_functions(), {"a", "b", "c"})

    def test_cycle_without_panic(self) -> None:
        # a -> b -> a (cycle), no panic anywhere.
        self.cg.add_call("a", "b")
        self.cg.add_call("b", "a")
        self.assertEqual(self.cg.find_panic_functions(), set())

    def test_isolated_functions(self) -> None:
        self.cg.add_function("isolated", panics=False)
        self.cg.add_function("panic_isolated", panics=True)
        self.assertEqual(self.cg.find_panic_functions(), {"panic_isolated"})

    def test_set_panic_after_edges(self) -> None:
        self.cg.add_call("main", "helper")
        self.cg.add_function("helper")
        self.cg.set_panic("helper")
        self.assertEqual(self.cg.find_panic_functions(), {"main", "helper"})

    def test_overwrite_panic_flag(self) -> None:
        self.cg.add_function("f", panics=True)
        self.cg.set_panic("f", panics=False)
        self.assertEqual(self.cg.find_panic_functions(), set())

    def test_complex_graph(self) -> None:
        # Build a more intricate graph with multiple layers and cross edges.
        edges = [
            ("A", "B"), ("A", "C"),
            ("B", "D"), ("C", "D"),
            ("D", "E"), ("E", "F"),
            ("F", "G"), ("G", "H"),
            ("H", "I"), ("I", "J"),
            ("J", "K"), ("K", "L"),
            ("L", "M"), ("M", "N"),
        ]
        for caller, callee in edges:
            self.cg.add_call(caller, callee)
        # Mark a few deep nodes as panic.
        self.cg.add_function("K", panics=True)
        self.cg.add_function("M", panics=True)
        # Expected panic set includes all ancestors of K and M.
        expected = {
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
            "L", "M"
        }
        self.assertEqual(self.cg.find_panic_functions(), expected)


def demo() -> None:
    """
    Simple demonstration printed to stdout.
    """
    cg = CallGraph()
    cg.add_call("handler", "parse")
    cg.add_call("handler", "validate")
    cg.add_call("validate", "check_limits")
    cg.add_function("check_limits", panics=True)
    cg.add_function("parse", panics=False)

    panic_funcs = cg.find_panic_functions()
    print("Functions that may panic:", sorted(panic_funcs))


if __name__ == "__main__":
    # Run unit tests.
    unittest.main(exit=False)

    # Run demo after tests.
    print("\n--- Demo ---")
    demo()
