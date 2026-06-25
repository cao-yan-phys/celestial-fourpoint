"""Spin-Gaunt and four-spin source-integral recoupling utilities."""

from __future__ import annotations

import sympy as sp

from ..invariant_tensors import wigner_3j
from .conventions import source_spin


PAIRINGS = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))


def spin_gaunt_prefactor(l1: int, l2: int, L: int, s1: int, s2: int, s3: int) -> sp.Expr:
    """Return the spin-only Gaunt prefactor used in the paired formula."""

    if s1 + s2 + s3 != 0:
        return sp.S.Zero
    prefactor = sp.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * L + 1) / (4 * sp.pi))
    return sp.simplify(prefactor * wigner_3j(l1, l2, L, -s1, -s2, -s3))


def _triangle_L_values(l1: int, l2: int, l3: int, l4: int) -> range:
    low = max(abs(l1 - l2), abs(l3 - l4))
    high = min(l1 + l2, l3 + l4)
    return range(low, high + 1)


def four_spin_integral_paired(
    ells: tuple[int, int, int, int],
    ms: tuple[int, int, int, int],
    spins: tuple[int, int, int, int],
    *,
    pairing: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> sp.Expr:
    """Evaluate the paired four-spin source integral formula."""

    if sum(spins) != 0:
        return sp.S.Zero
    i1, i2, i3, i4 = pairing
    l1, l2, l3, l4 = (ells[i1], ells[i2], ells[i3], ells[i4])
    m1, m2, m3, m4 = (ms[i1], ms[i2], ms[i3], ms[i4])
    s1, s2, s3, s4 = (spins[i1], spins[i2], spins[i3], spins[i4])
    S = s1 + s2
    if S + s3 + s4 != 0:
        return sp.S.Zero

    total = sp.S.Zero
    for L in _triangle_L_values(l1, l2, l3, l4):
        g12 = spin_gaunt_prefactor(l1, l2, L, s1, s2, -S)
        g34 = spin_gaunt_prefactor(l3, l4, L, s3, s4, S)
        if g12 == 0 or g34 == 0:
            continue
        for M in range(-L, L + 1):
            term = (
                sp.Integer(-1) ** (M + S)
                * g12
                * g34
                * wigner_3j(l1, l2, L, m1, m2, M)
                * wigner_3j(l3, l4, L, m3, m4, -M)
            )
            total += term
    return sp.simplify(total)


def source_spins_from_epsilons(epsilons: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Return source-side spins for a four-point helicity assignment."""

    return tuple(source_spin(eps) for eps in epsilons)


def pairing_spin_cost(epsilons: tuple[int, int, int, int], pairing: tuple[int, int, int, int]) -> int:
    """Return |S| for the first pair in a source-side pairing."""

    spins = source_spins_from_epsilons(epsilons)
    i1, i2, _i3, _i4 = pairing
    return abs(spins[i1] + spins[i2])


def choose_min_spin_pairing(epsilons: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Choose the source pairing that minimizes the intermediate spin |S|."""

    return min(PAIRINGS, key=lambda pairing: pairing_spin_cost(epsilons, pairing))
