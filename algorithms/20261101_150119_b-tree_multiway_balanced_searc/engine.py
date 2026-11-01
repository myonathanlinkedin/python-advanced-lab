from __future__ import annotations
from typing import List, Tuple, TypeVar, Generic
from custom_types import BTreeNode

T = TypeVar('T')


def split_child(parent: BTreeNode[T], index: int) -> Tuple[T, BTreeNode[T]]:
    """
    Split the full child node at parent.children[index] into two nodes.
    The middle key is promoted to the parent.

    Parameters
    ----------
    parent : BTreeNode[T]
        The node whose child is to be split. Must not be a leaf.
    index : int
        Position of the child to split (0 <= index < len(parent.children)).

    Returns
    -------
    Tuple[T, BTreeNode[T]]
        (promoted_key, new_right_node) where `new_right_node` is the node
        containing the keys greater than the promoted key.
    """
    child = parent.children[index]
    assert child.is_full(), "split_child called on a non-full child"

    t = child.t
    # Create new node that will hold the right half of child's keys
    right = BTreeNode(t=t, leaf=child.leaf)

    # Keys: left child keeps first t-1 keys, right gets last t-1 keys
    right.keys = child.keys[t:]                # keys[t] .. keys[2t-2]
    promoted_key = child.keys[t - 1]           # middle key

    # Trim left child keys
    child.keys = child.keys[:t - 1]

    if not child.leaf:
        # Children are split similarly: left keeps first t children, right gets remaining t
        right.children = child.children[t:]   # children[t] .. children[2t-1]
        child.children = child.children[:t]   # first t children

    # Insert promoted key and new right child into parent
    parent.keys.insert(index, promoted_key)
    parent.children.insert(index + 1, right)

    return promoted_key, right


def insert_non_full(node: BTreeNode[T], key: T) -> None:
    """
    Insert a key into a node that is guaranteed not to be full.
    This is a helper for the B-Tree insertion algorithm.
    """
    i = len(node.keys) - 1
    if node.leaf:
        # Insert key into the correct position within the leaf
        node.keys.append(None)  # placeholder for shifting
        while i >= 0 and key < node.keys[i]:
            node.keys[i + 1] = node.keys[i]
            i -= 1
        node.keys[i + 1] = key
    else:
        # Find child to descend into
        while i >= 0 and key < node.keys[i]:
            i -= 1
        i += 1
        child = node.children[i]
        if child.is_full():
            split_child(node, i)
            # After split, decide which of the two children to descend into
            if key > node.keys[i]:
                i += 1
        insert_non_full(node.children[i], key)


class BTree(Generic[T]):
    """
    Minimal B-Tree implementation exposing insertion and internal state.
    Only the split operation is required for the assignment, but a full
    insertion routine is provided for demonstration and testing.
    """
    def __init__(self, t: int):
        assert t >= 2, "Minimum degree t must be at least 2"
        self.t = t
        self.root = BTreeNode(t=t, leaf=True)

    def insert(self, key: T) -> None:
        root = self.root
        if root.is_full():
            # Create new root and split old root
            new_root = BTreeNode(t=self.t, leaf=False, children=[root])
            split_child(new_root, 0)
            self.root = new_root
            insert_non_full(new_root, key)
        else:
            insert_non_full(root, key)

    def __repr__(self) -> str:
        return f"BTree(t={self.t}, root={self.root})"
