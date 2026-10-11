from __future__ import annotations
from typing import Any, List, Optional, Callable, Dict, Tuple


class TreeNode:
    """
    Simple binary tree node.
    """
    __slots__ = ("value", "left", "right")

    def __init__(self, value: Any,
                 left: Optional[TreeNode] = None,
                 right: Optional[TreeNode] = None) -> None:
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"TreeNode({self.value!r})"


def _recursive_preorder(node: Optional[TreeNode], out: List[Any]) -> None:
    if node is None:
        return
    out.append(node.value)
    _recursive_preorder(node.left, out)
    _recursive_preorder(node.right, out)


def recursive_preorder(root: Optional[TreeNode]) -> List[Any]:
    """Return preorder traversal (root, left, right) using recursion."""
    result: List[Any] = []
    _recursive_preorder(root, result)
    return result


def _recursive_inorder(node: Optional[TreeNode], out: List[Any]) -> None:
    if node is None:
        return
    _recursive_inorder(node.left, out)
    out.append(node.value)
    _recursive_inorder(node.right, out)


def recursive_inorder(root: Optional[TreeNode]) -> List[Any]:
    """Return inorder traversal (left, root, right) using recursion."""
    result: List[Any] = []
    _recursive_inorder(root, result)
    return result


def _recursive_postorder(node: Optional[TreeNode], out: List[Any]) -> None:
    if node is None:
        return
    _recursive_postorder(node.left, out)
    _recursive_postorder(node.right, out)
    out.append(node.value)


def recursive_postorder(root: Optional[TreeNode]) -> List[Any]:
    """Return postorder traversal (left, right, root) using recursion."""
    result: List[Any] = []
    _recursive_postorder(root, result)
    return result


def iterative_preorder(root: Optional[TreeNode]) -> List[Any]:
    """Iterative preorder using an explicit stack."""
    if root is None:
        return []
    stack: List[TreeNode] = [root]
    result: List[Any] = []
    while stack:
        node = stack.pop()
        result.append(node.value)
        # push right first so left is processed first
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return result


def iterative_inorder(root: Optional[TreeNode]) -> List[Any]:
    """Iterative inorder using a stack and a pointer."""
    result: List[Any] = []
    stack: List[TreeNode] = []
    current = root
    while stack or current:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.value)
        current = current.right
    return result


def iterative_postorder(root: Optional[TreeNode]) -> List[Any]:
    """Iterative postorder using two stacks (classic approach)."""
    if root is None:
        return []
    stack1: List[TreeNode] = [root]
    stack2: List[TreeNode] = []
    while stack1:
        node = stack1.pop()
        stack2.append(node)
        if node.left is not None:
            stack1.append(node.left)
        if node.right is not None:
            stack1.append(node.right)
    # stack2 now holds nodes in reverse postorder
    return [node.value for node in reversed(stack2)]


def traversal_matrix(root: Optional[TreeNode]) -> Dict[str, Tuple[List[Any], List[Any]]]:
    """
    Produce a matrix (dictionary) where each key is a traversal name and the value
    is a tuple (recursive_result, iterative_result). The function asserts that
    both results are identical; otherwise it raises AssertionError.
    """
    traversals: Dict[str, Callable[[Optional[TreeNode]], List[Any]]] = {
        "preorder": recursive_preorder,
        "inorder": recursive_inorder,
        "postorder": recursive_postorder,
    }
    iterative_funcs: Dict[str, Callable[[Optional[TreeNode]], List[Any]]] = {
        "preorder": iterative_preorder,
        "inorder": iterative_inorder,
        "postorder": iterative_postorder,
    }
    matrix: Dict[str, Tuple[List[Any], List[Any]]] = {}
    for name, rec_func in traversals.items():
        rec_res = rec_func(root)
        itr_res = iterative_funcs[name](root)
        assert rec_res == itr_res, f"{name} mismatch: {rec_res} != {itr_res}"
        matrix[name] = (rec_res, itr_res)
    return matrix


__all__ = [
    "TreeNode",
    "recursive_preorder",
    "recursive_inorder",
    "recursive_postorder",
    "iterative_preorder",
    "iterative_inorder",
    "iterative_postorder",
    "traversal_matrix",
]
