import unittest
import time
from typing import Dict, Any, Set

from core import Component, PipeLinker


def make_component(
    name: str,
    inputs: Set[str],
    outputs: Set[str],
    operation: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Component:
    """
    Helper to create a Component from a simple operation.
    """
    return Component(name=name, func=operation, inputs=inputs, outputs=outputs)


# Sample kernel‑like operations -------------------------------------------------
def op_add_one(data: Dict[str, Any]) -> Dict[str, Any]:
    # Expects 'x', produces 'y' = x + 1
    return {"y": data["x"] + 1}


def op_mul_two(data: Dict[str, Any]) -> Dict[str, Any]:
    # Expects 'y', produces 'z' = y * 2
    return {"z": data["y"] * 2}


def op_concat(data: Dict[str, Any]) -> Dict[str, Any]:
    # Expects 'z' and 'suffix', produces 'w' = str(z) + suffix
    return {"w": f"{data['z']}{data['suffix']}"}


# Unit tests --------------------------------------------------------------------
class TestPipeLinker(unittest.TestCase):
    def setUp(self) -> None:
        self.comp_a = make_component(
            "AddOne", inputs={"x"}, outputs={"y"}, operation=op_add_one
        )
        self.comp_b = make_component(
            "MulTwo", inputs={"y"}, outputs={"z"}, operation=op_mul_two
        )
        self.comp_c = make_component(
            "Concat", inputs={"z", "suffix"}, outputs={"w"}, operation=op_concat
        )

    def test_successful_link_and_execution(self) -> None:
        linker = PipeLinker([self.comp_a, self.comp_b, self.comp_c])
        initial = {"x": 3, "suffix": "_end"}
        result = linker.execute(initial)
        expected = {"x": 3, "suffix": "_end", "y": 4, "z": 8, "w": "8_end"}
        self.assertEqual(result, expected)

    def test_missing_initial_input(self) -> None:
        linker = PipeLinker([self.comp_a, self.comp_b])
        initial = {"suffix": "oops"}  # 'x' missing
        with self.assertRaises(KeyError):
            linker.execute(initial)

    def test_contract_violation_between_components(self) -> None:
        # Intentionally break the contract: comp_c needs 'z' but we omit comp_b.
        linker = PipeLinker([self.comp_a, self.comp_c])
        initial = {"x": 5, "suffix": "x"}
        with self.assertRaises(ValueError):
            linker.link(set(initial.keys()))

    def test_circular_dependency_detection(self) -> None:
        # Create two components that each require the other's output.
        comp1 = make_component(
            "C1", inputs={"b"}, outputs={"a"}, operation=lambda d: {"a": d["b"] + 1}
        )
        comp2 = make_component(
            "C2", inputs={"a"}, outputs={"b"}, operation=lambda d: {"b": d["a"] * 2}
        )
        linker = PipeLinker([comp1, comp2])
        with self.assertRaises(ValueError):
            linker.link(set())

    def test_validate_chain_explicit_order(self) -> None:
        linker = PipeLinker([self.comp_a, self.comp_b, self.comp_c])
        order = [self.comp_a, self.comp_b, self.comp_c]
        self.assertTrue(linker.validate_chain(order, {"x", "suffix"}))

    def test_validate_chain_wrong_order(self) -> None:
        linker = PipeLinker([self.comp_a, self.comp_b, self.comp_c])
        order = [self.comp_c, self.comp_a, self.comp_b]
        with self.assertRaises(ValueError):
            linker.validate_chain(order, {"x", "suffix"})

    def test_component_output_mismatch(self) -> None:
        # Component declares output 'y' but returns nothing.
        bad = make_component(
            "Bad",
            inputs={"x"},
            outputs={"y"},
            operation=lambda d: {},  # missing 'y'
        )
        linker = PipeLinker([bad])
        with self.assertRaises(ValueError):
            linker.execute({"x": 1})

    def test_performance_small_pipeline(self) -> None:
        # Simple benchmark to ensure deterministic runtime (< 0.01s)
        linker = PipeLinker([self.comp_a, self.comp_b, self.comp_c])
        start = time.perf_counter()
        for i in range(1000):
            linker.execute({"x": i, "suffix": "_s"})
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.5)  # generous bound for CI environments


if __name__ == "__main__":
    unittest.main()
