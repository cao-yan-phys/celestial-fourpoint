"""Wigner symbols and invariant tensor contractions."""

from __future__ import annotations

from functools import cache

import sympy as sp
from sympy.physics.wigner import clebsch_gordan as _clebsch_gordan
from sympy.physics.wigner import wigner_3j as _wigner_3j

from .coupling_trees import Leaf, Node, Tree


@cache
def wigner_3j(l1: int, l2: int, l3: int, m1: int, m2: int, m3: int) -> sp.Expr:
    """Cached exact Wigner 3j symbol."""

    return _wigner_3j(l1, l2, l3, m1, m2, m3)


@cache
def clebsch_gordan(l1: int, m1: int, l2: int, m2: int, L: int, M: int) -> sp.Expr:
    """Cached exact Clebsch-Gordan coefficient."""

    return _clebsch_gordan(l1, l2, L, m1, m2, M)


def three_point_invariant(l1: int, l2: int, l3: int, m1: int, m2: int, m3: int) -> sp.Expr:
    """Rotational invariant for a three-point harmonic tensor."""

    return wigner_3j(l1, l2, l3, m1, m2, m3)


def four_point_invariant_channel(
    l1: int,
    l2: int,
    l3: int,
    l4: int,
    m1: int,
    m2: int,
    m3: int,
    m4: int,
    L: int,
    *,
    include_root_cg: bool = False,
) -> sp.Expr:
    """Four-point invariant in the ((12)(34)) channel.

    By default this returns sum_M C(12->LM) C(34->L,-M), matching the
    minimal object in the implementation guide. With ``include_root_cg=True``
    it includes the final C(LM, L-M -> 00) scalar coupling.
    """

    total = sp.S.Zero
    for M in range(-L, L + 1):
        term = clebsch_gordan(l1, m1, l2, m2, L, M) * clebsch_gordan(
            l3, m3, l4, m4, L, -M
        )
        if include_root_cg:
            term *= clebsch_gordan(L, M, L, -M, 0, 0)
        total += term
    return sp.simplify(total)


def _resolve_label(label: str | int | None, internal_ells: dict[str | int, int]) -> int:
    if label is None:
        raise ValueError("each non-leaf node needs an internal angular label")
    if isinstance(label, int):
        return label
    if label not in internal_ells:
        raise KeyError(f"missing internal angular momentum for {label!r}")
    return internal_ells[label]


def _subtree_states(
    tree: Tree,
    ells: tuple[int, ...],
    ms: tuple[int, ...],
    internal_ells: dict[str | int, int],
) -> dict[tuple[int, int], sp.Expr]:
    if isinstance(tree, Leaf):
        idx = tree.index - 1
        return {(ells[idx], ms[idx]): sp.S.One}

    left_states = _subtree_states(tree.left, ells, ms, internal_ells)
    right_states = _subtree_states(tree.right, ells, ms, internal_ells)
    L = _resolve_label(tree.internal_label, internal_ells)
    out: dict[tuple[int, int], sp.Expr] = {}
    for (l_left, m_left), coeff_left in left_states.items():
        for (l_right, m_right), coeff_right in right_states.items():
            M = m_left + m_right
            if abs(M) > L:
                continue
            coeff = (
                coeff_left
                * coeff_right
                * clebsch_gordan(l_left, m_left, l_right, m_right, L, M)
            )
            out[(L, M)] = out.get((L, M), sp.S.Zero) + coeff
    return {key: sp.simplify(value) for key, value in out.items() if value != 0}


def tree_invariant(
    tree: Tree,
    ells: tuple[int, ...],
    ms: tuple[int, ...],
    internal_ells: dict[str | int, int],
) -> sp.Expr:
    """Evaluate a scalar invariant from a coupling tree."""

    if len(ells) != len(ms):
        raise ValueError("ells and ms must have the same length")
    states = _subtree_states(tree, tuple(ells), tuple(ms), dict(internal_ells))
    return sp.simplify(states.get((0, 0), sp.S.Zero))
