"""Wigner-6j recoupling helpers for four-point blocks."""

from __future__ import annotations

from functools import cache

import sympy as sp
from sympy.physics.wigner import wigner_6j as _wigner_6j

from .coupled_blocks import coupled_block_m_coefficient, intermediate_L_values


@cache
def wigner_6j(j1: int, j2: int, j3: int, j4: int, j5: int, j6: int) -> sp.Expr:
    """Cached exact Wigner 6j symbol."""

    return _wigner_6j(j1, j2, j3, j4, j5, j6)


def recoupling_12_34_to_13_24(
    l1: int,
    l2: int,
    l3: int,
    l4: int,
    L: int,
    Lp: int,
) -> sp.Expr:
    """Return the 6j recoupling coefficient in this block normalization.

    For the unnormalized m-space block

    B_L(12|34) = sum_M (-1)^M (l1 l2 L; m1 m2 M)
                         (l3 l4 L; m3 m4 -M),

    direct inner-product calibration gives

    <B_L(12|34), B_L'(13|24)>
      = (-1)^(l2+l3) {l1 l2 L; l4 l3 L'}.
    """

    phase = sp.Integer(-1) ** (l2 + l3)
    return sp.simplify(phase * wigner_6j(l1, l2, L, l4, l3, Lp))


def recoupling_inner_product_12_34_to_13_24(
    l1: int,
    l2: int,
    l3: int,
    l4: int,
    L: int,
    Lp: int,
) -> sp.Expr:
    """Compute the exact m-space inner product between two channel blocks."""

    ells = (l1, l2, l3, l4)
    if L not in intermediate_L_values(ells, (0, 1, 2, 3)):
        return sp.S.Zero
    if Lp not in intermediate_L_values(ells, (0, 2, 1, 3)):
        return sp.S.Zero
    total = sp.S.Zero
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            for m3 in range(-l3, l3 + 1):
                for m4 in range(-l4, l4 + 1):
                    ms = (m1, m2, m3, m4)
                    total += coupled_block_m_coefficient(
                        ells, L, ms, pairing=(0, 1, 2, 3)
                    ) * coupled_block_m_coefficient(ells, Lp, ms, pairing=(0, 2, 1, 3))
    return sp.simplify(total)


def verify_wigner_6j_recoupling(
    l1: int,
    l2: int,
    l3: int,
    l4: int,
    L: int,
    Lp: int,
) -> bool:
    """Check the recoupling coefficient against direct m-space inner product."""

    return (
        sp.simplify(
            recoupling_inner_product_12_34_to_13_24(l1, l2, l3, l4, L, Lp)
            - recoupling_12_34_to_13_24(l1, l2, l3, l4, L, Lp)
        )
        == 0
    )
