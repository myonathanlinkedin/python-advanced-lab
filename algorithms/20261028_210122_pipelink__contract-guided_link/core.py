from __future__ import annotations

from typing import Callable, Dict, Any, Set, List, Tuple


class Component:
    """
    Represents a GPU kernel component (simulated as a Python callable) with
    explicit input and output contracts.
    """

    def __init__(
        self,
        name: str,
        func: Callable[[Dict[str, Any]], Dict[str, Any]],
        inputs: Set[str],
        outputs: Set[str],
    ) -> None:
        self.name = name
        self.func = func
        self.inputs = set(inputs)
        self.outputs = set(outputs)

    def __repr__(self) -> str:
        return f"Component({self.name!r})"

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the component on the supplied data dictionary.

        Raises:
            KeyError: if required inputs are missing.
            ValueError: if declared outputs are not produced.
        Returns:
            Updated data dictionary containing new outputs.
        """
        missing = self.inputs - data.keys()
        if missing:
            raise KeyError(
                f"Component {self.name} missing inputs: {sorted(missing)}"
            )
        # Execute the kernel simulation.
        result = self.func(data)
        if not isinstance(result, dict):
            raise TypeError(
                f"Component {self.name} must return a dict, got {type(result)}"
            )
        missing_out = self.outputs - result.keys()
        if missing_out:
            raise ValueError(
                f"Component {self.name} did not produce declared outputs: {sorted(missing_out)}"
            )
        # Merge result into data (overwrites allowed).
        data.update(result)
        return data


class PipeLinker:
    """
    Links a collection of Components into a contract‑compatible pipeline.
    """

    def __init__(self, components: List[Component]) -> None:
        self.components = list(components)

    def _is_compatible(
        self, comp: Component, available: Set[str]
    ) -> bool:
        """Check whether a component's inputs are satisfied by `available` keys."""
        return comp.inputs.issubset(available)

    def link(self, initial_keys: Set[str]) -> List[Component]:
        """
        Determine an execution order that respects all contracts.

        Args:
            initial_keys: Keys present before any component runs.

        Returns:
            Ordered list of components.

        Raises:
            ValueError: If no valid ordering exists (e.g., circular dependency).
        """
        remaining = set(self.components)
        order: List[Component] = []
        available = set(initial_keys)

        while remaining:
            progress = False
            for comp in list(remaining):
                if self._is_compatible(comp, available):
                    order.append(comp)
                    available.update(comp.outputs)
                    remaining.remove(comp)
                    progress = True
            if not progress:
                # No component could be placed – contract violation.
                missing_info = {
                    comp.name: sorted(comp.inputs - available) for comp in remaining
                }
                raise ValueError(
                    f"Cannot resolve contracts; remaining components: {missing_info}"
                )
        return order

    def validate_chain(self, order: List[Component], initial_keys: Set[str]) -> bool:
        """
        Verify that a given ordering satisfies all contracts.

        Returns:
            True if valid, otherwise raises an exception.
        """
        available = set(initial_keys)
        for comp in order:
            if not self._is_compatible(comp, available):
                raise ValueError(
                    f"Component {comp.name} cannot run; missing inputs {sorted(comp.inputs - available)}"
                )
            available.update(comp.outputs)
        return True

    def execute(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the linked pipeline on `initial_data`.

        Returns:
            Final data dictionary after all components have executed.
        """
        order = self.link(set(initial_data.keys()))
        data = dict(initial_data)  # shallow copy
        for comp in order:
            data = comp.run(data)
        return data


__all__ = ["Component", "PipeLinker"]
