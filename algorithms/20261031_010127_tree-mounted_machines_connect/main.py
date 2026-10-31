from __future__ import annotations

from engine import TreeMachineNetwork


def _run_unit_tests() -> None:
    # Initialise network with default root
    net = TreeMachineNetwork()
    assert net.get_knock_count("root") == 0, "Root should start with zero knocks."

    # Add a small hierarchy
    net.add_node("branch1", "root")
    net.add_node("branch2", "root")
    net.add_node("leaf1", "branch1")
    net.add_node("leaf2", "branch1")
    net.add_node("leaf3", "branch2")

    # Verify structure via internal state (no direct access, use API)
    assert net.get_node("branch1").parent_id == "root"
    assert net.get_node("leaf2").parent_id == "branch1"
    assert net.get_node("root").children == {"branch1", "branch2"}

    # Perform knocks
    net.knock("root")
    net.knock("branch1")
    net.knock("leaf2")
    net.knock("leaf2")
    net.knock("leaf3")
    net.knock("leaf3")
    net.knock("leaf3")

    # Validate individual counts
    assert net.get_knock_count("root") == 1
    assert net.get_knock_count("branch1") == 1
    assert net.get_knock_count("leaf2") == 2
    assert net.get_knock_count("leaf3") == 3

    # Validate subtree aggregates
    assert net.total_knocks_subtree("root") == 7  # sum of all nodes
    assert net.total_knocks_subtree("branch1") == 4  # branch1 + leaf1(0) + leaf2(2)
    assert net.total_knocks_subtree("leaf2") == 2

    # Edge case: knocking unknown node raises KeyError
    try:
        net.knock("nonexistent")
        assert False, "Expected KeyError for unknown node."
    except KeyError:
        pass

    # Edge case: adding duplicate node raises ValueError
    try:
        net.add_node("leaf1", "branch2")
        assert False, "Expected ValueError for duplicate node id."
    except ValueError:
        pass

    # Edge case: removing root raises ValueError
    try:
        net.remove_node("root")
        assert False, "Expected ValueError when removing root."
    except ValueError:
        pass

    # Remove a subtree and verify counts adjust
    net.remove_node("branch1")
    assert "branch1" not in net._nodes
    assert "leaf1" not in net._nodes
    assert "leaf2" not in net._nodes
    # Remaining knocks should be from root and branch2 subtree
    assert net.total_knocks_subtree("root") == 4  # root(1) + branch2(0) + leaf3(3)

    # Reset all counters and verify zeroed state
    net.reset_all()
    for nid in net._nodes:
        assert net.get_knock_count(nid) == 0, f"Node {nid} should be reset to zero."

    print("All unit tests passed.")


def _demo_usage() -> None:
    """
    Simple interactive demonstration (no I/O, just procedural steps).
    """
    net = TreeMachineNetwork()
    net.add_node("north", "root")
    net.add_node("south", "root")
    net.add_node("east", "north")
    net.add_node("west", "north")

    # Simulate a user knocking on the "west" machine three times
    for _ in range(3):
        net.knock("west")

    # Show aggregated knocks for the "north" branch
    north_total = net.total_knocks_subtree("north")
    assert north_total == 3, "North subtree should have three knocks."

    # Demonstrate removal of a leaf
    net.remove_node("east")
    assert "east" not in net._nodes

    # Final state summary (purely for verification)
    summary = {nid: net.get_knock_count(nid) for nid in net._nodes}
    expected = {"root": 0, "north": 0, "south": 0, "west": 3}
    assert summary == expected, "Final state does not match expected counts."

    print("Demo completed successfully.")


if __name__ == "__main__":
    _run_unit_tests()
    _demo_usage()
