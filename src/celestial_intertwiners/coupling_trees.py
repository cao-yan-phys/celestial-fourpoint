"""Binary coupling trees for rotationally invariant tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class Leaf:
    """External one-based leaf index."""

    index: int

    def __post_init__(self):
        if self.index < 1:
            raise ValueError("leaf indices are one-based and must be positive")


@dataclass(frozen=True)
class Node:
    """Binary coupling node."""

    left: "Tree"
    right: "Tree"
    internal_label: str | int | None = None


Tree: TypeAlias = Leaf | Node


def leaves(tree: Tree) -> tuple[int, ...]:
    """Return leaf indices from left to right."""

    if isinstance(tree, Leaf):
        return (tree.index,)
    return leaves(tree.left) + leaves(tree.right)


def internal_labels(tree: Tree) -> tuple[str | int | None, ...]:
    """Return internal labels from a tree."""

    if isinstance(tree, Leaf):
        return ()
    return internal_labels(tree.left) + internal_labels(tree.right) + (tree.internal_label,)


def default_four_point_tree() -> Node:
    """Return the ((12)(34)) -> 0 tree."""

    return Node(
        Node(Leaf(1), Leaf(2), internal_label="L12"),
        Node(Leaf(3), Leaf(4), internal_label="L34"),
        internal_label=0,
    )


def validate_tree(tree: Tree, n_leaves: int | None = None) -> None:
    """Validate one-based leaves and, optionally, the expected leaf count."""

    leaf_indices = leaves(tree)
    if len(set(leaf_indices)) != len(leaf_indices):
        raise ValueError("coupling tree contains duplicate leaves")
    if n_leaves is not None and set(leaf_indices) != set(range(1, n_leaves + 1)):
        raise ValueError("coupling tree leaves do not match 1..n")
