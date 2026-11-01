from __future__ import annotations
from typing import List
from custom_types import BTreeNode, BTree
from engine import split_child

def test_split_leaf_node() -> None:
    """
    Verify that splitting a full leaf node produces two leaf nodes
    with t-1 keys each and promotes the middle key.
    """
    t = 3
    # Create a full leaf node with 2*t-1 = 5 keys
    leaf = BTreeNode[int](t=t, keys=[1, 2, 3, 4, 5], leaf=True)
    parent = BTreeNode[int](t=t, keys=[], children=[leaf], leaf=False)

    promoted, right = split_child(parent, 0)

    # Expected state
    assert promoted == 3, f"Promoted key should be 3, got {promoted}"
    assert parent.keys == [3], f"Parent keys incorrect: {parent.keys}"
    assert len(parent.children) == 2, "Parent should have two children after split"
    assert leaf.keys == [1, 2], f"Left child keys incorrect: {leaf.keys}"
    assert right.keys == [4, 5], f"Right child keys incorrect: {right.keys}"
    assert leaf.leaf and right.leaf, "Both children must remain leaves"


def test_split_internal_node() -> None:
    """
    Verify splitting an internal node correctly redistributes children.
    """
    t = 2
    # Full internal node with 3 keys and 4 children
    child0 = BTreeNode[int](t=t, keys=[1], leaf=True)
    child1 = BTreeNode[int](t=t, keys=[3], leaf=True)
    child2 = BTreeNode[int](t=t, keys=[5], leaf=True)
    child3 = BTreeNode[int](t=t, keys=[7], leaf=True)
    internal = BTreeNode[int](
        t=t,
        keys=[2, 4, 6],
        children=[child0, child1, child2, child3],
        leaf=False,
    )
    parent = BTreeNode[int](t=t, keys=[], children=[internal], leaf=False)

    promoted, right = split_child(parent, 0)

    # After split, internal (left) should have first key [2] and two children
    assert internal.keys == [2], f"Left internal keys incorrect: {internal.keys}"
    assert len(internal.children) == 2, "Left internal should have 2 children"
    assert internal.children[0] is child0 and internal.children[1] is child1

    # Right node should have last key [6] and two children
    assert right.keys == [6], f"Right internal keys incorrect: {right.keys}"
    assert len(right.children) == 2, "Right internal should have 2 children"
    assert right.children[0] is child2 and right.children[1] is child3

    # Parent should have promoted key 4
    assert parent.keys == [4], f"Parent keys incorrect: {parent.keys}"
    assert len(parent.children) == 2, "Parent should have two children after split"


def test_btree_insertion_and_multiple_splits() -> None:
    """
    Insert a sequence of keys that forces several splits, including root splits.
    Verify B-Tree invariants after each insertion.
    """
    t = 3
    tree = BTree[int](t=t)
    keys_to_insert = [20, 10, 30, 5, 15, 25, 35, 1, 6, 12, 18, 22, 28, 32, 38]

    for key in keys_to_insert:
        tree.insert(key)

    # After all insertions, the tree should have height 2 (root + leaves)
    assert not tree.root.leaf, "Root must be internal after many insertions"
    assert len(tree.root.keys) == 2, f"Root should have 2 keys, got {len(tree.root.keys)}"
    # Verify that all leaf nodes have at most 2*t-1 keys
    def check_node(node: BTreeNode[int]) -> None:
        assert len(node.keys) <= 2 * t - 1, "Node exceeds maximum key count"
        if not node.leaf:
            for child in node.children:
                check_node(child)
    check_node(tree.root)


def run_all_tests() -> None:
    test_split_leaf_node()
    test_split_internal_node()
    test_btree_insertion_and_multiple_splits()
    print("All B-Tree splitter tests passed.")


if __name__ == '__main__':
    run_all_tests()
