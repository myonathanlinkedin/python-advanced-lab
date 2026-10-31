from __future__ import annotations

from typing import Dict, List

from custom_types import TreeNode


class TreeMachineNetwork:
    """
    Core algorithmic implementation managing a rooted tree of TreeNode objects.
    Provides O(1) node lookup and O(k) operations where k is the size of a subtree.
    """

    def __init__(self, root_id: str = "root") -> None:
        """
        Initialise the network with a single root node.
        :param root_id: Identifier for the root machine.
        """
        root = TreeNode(node_id=root_id, parent_id=None)
        self._nodes: Dict[str, TreeNode] = {root_id: root}
        self._root_id = root_id

    # --------------------------------------------------------------------- #
    # Node management
    # --------------------------------------------------------------------- #
    def add_node(self, node_id: str, parent_id: str) -> None:
        """
        Add a new node under the given parent.
        :raises ValueError: If node_id already exists or parent_id is unknown.
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists.")
        if parent_id not in self._nodes:
            raise ValueError(f"Parent node '{parent_id}' does not exist.")
        parent = self._nodes[parent_id]
        new_node = TreeNode(node_id=node_id, parent_id=parent_id)
        parent.children.add(node_id)
        self._nodes[node_id] = new_node

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and its entire subtree from the network.
        :raises ValueError: If attempting to remove the root or unknown node.
        """
        if node_id == self._root_id:
            raise ValueError("Cannot remove the root node.")
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' does not exist.")
        # Gather all nodes to delete (subtree traversal)
        to_delete = self._collect_subtree(node_id)
        # Detach from parent
        parent_id = self._nodes[node_id].parent_id
        if parent_id is not None:
            self._nodes[parent_id].children.discard(node_id)
        # Delete nodes
        for nid in to_delete:
            del self._nodes[nid]

    def get_node(self, node_id: str) -> TreeNode:
        """Retrieve a node object; raises KeyError if missing."""
        return self._nodes[node_id]

    # --------------------------------------------------------------------- #
    # Knock operations
    # --------------------------------------------------------------------- #
    def knock(self, node_id: str) -> None:
        """
        Register a knock on the specified node.
        :raises KeyError: If node_id does not exist.
        """
        self._nodes[node_id].knock()

    def get_knock_count(self, node_id: str) -> int:
        """Return the knock count for a single node."""
        return self._nodes[node_id].knock_count

    def total_knocks_subtree(self, node_id: str) -> int:
        """
        Compute the sum of knock counts for the node and all its descendants.
        Runs in O(k) where k is the size of the subtree.
        """
        total = 0
        for nid in self._collect_subtree(node_id):
            total += self._nodes[nid].knock_count
        return total

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def _collect_subtree(self, node_id: str) -> List[str]:
        """
        Depth‑first collection of all node identifiers in the subtree rooted at node_id.
        Returns a list ordered by pre‑order traversal.
        """
        stack = [node_id]
        collected: List[str] = []
        while stack:
            current = stack.pop()
            collected.append(current)
            # Add children to stack; order does not matter for correctness
            stack.extend(self._nodes[current].children)
        return collected

    def reset_all(self) -> None:
        """Reset knock counters for every node in the network."""
        for node in self._nodes.values():
            node.reset()
