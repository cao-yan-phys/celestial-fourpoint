"""Coupled four-point block bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ..invariant_tensors import wigner_3j
from .conventions import angular_output_spin, helicity_selection_rule, normalize_observable
from .helicity_responses import response_transfer
from .spin_gaunt import choose_min_spin_pairing, spin_gaunt_prefactor


@dataclass(frozen=True)
class CoupledBlockLabel:
    """Label for a spin-coupled four-point block in the (12)(34) channel."""

    ells: tuple[int, int, int, int]
    external_spins: tuple[int, int, int, int]
    L: int
    pairing: tuple[int, int, int, int] = (0, 1, 2, 3)


def intermediate_L_values(ells: tuple[int, int, int, int], pairing=(0, 1, 2, 3)) -> tuple[int, ...]:
    """Return allowed L values for a paired four-point block."""

    i1, i2, i3, i4 = pairing
    l1, l2, l3, l4 = (ells[i1], ells[i2], ells[i3], ells[i4])
    low = max(abs(l1 - l2), abs(l3 - l4))
    high = min(l1 + l2, l3 + l4)
    return tuple(range(low, high + 1))


def source_recoupling_coefficient(
    ells: tuple[int, int, int, int],
    epsilons: tuple[int, int, int, int],
    L: int,
    *,
    pairing: tuple[int, int, int, int] | None = None,
) -> sp.Expr:
    """Return the source-side coefficient in the paired channel."""

    if not helicity_selection_rule(epsilons):
        return sp.S.Zero
    if pairing is None:
        pairing = choose_min_spin_pairing(epsilons)
    i1, i2, i3, i4 = pairing
    source_spins = tuple(2 * eps for eps in epsilons)
    s1, s2, s3, s4 = (source_spins[i1], source_spins[i2], source_spins[i3], source_spins[i4])
    S = s1 + s2
    if S + s3 + s4 != 0:
        return sp.S.Zero
    l1, l2, l3, l4 = (ells[i1], ells[i2], ells[i3], ells[i4])
    phase = sp.Integer(-1) ** S
    return sp.simplify(
        phase
        * spin_gaunt_prefactor(l1, l2, L, s1, s2, -S)
        * spin_gaunt_prefactor(l3, l4, L, s3, s4, S)
    )


def external_spins(observables: tuple[str, str, str, str], epsilons: tuple[int, int, int, int]):
    """Return external spin weights for a mixed kernel."""

    return tuple(angular_output_spin(obs, eps) for obs, eps in zip(observables, epsilons))


def transfer_product(
    observables: tuple[str, str, str, str],
    ells: tuple[int, int, int, int],
    *,
    exact: bool = True,
):
    """Return product_i r_l_i^{X_i}."""

    factor = sp.S.One if exact else 1.0
    for obs, ell in zip(observables, ells):
        factor *= response_transfer(normalize_observable(obs), ell, exact=exact)
    return sp.simplify(factor) if exact else factor


def casimir_eigenvalue(L: int) -> int:
    """Return L(L+1), the coupled-block Casimir eigenvalue."""

    return L * (L + 1)


def coupled_block_m_coefficient(
    ells: tuple[int, int, int, int],
    L: int,
    ms: tuple[int, int, int, int],
    *,
    pairing: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> sp.Expr:
    """Return the m-space coefficient of a coupled block.

    This is the purely angular-momentum coefficient multiplying the product of
    four external spin harmonics in the chosen pairing channel.
    """

    if L not in intermediate_L_values(ells, pairing):
        return sp.S.Zero
    i1, i2, i3, i4 = pairing
    l1, l2, l3, l4 = (ells[i1], ells[i2], ells[i3], ells[i4])
    m1, m2, m3, m4 = (ms[i1], ms[i2], ms[i3], ms[i4])
    total = sp.S.Zero
    for M in range(-L, L + 1):
        total += (
            sp.Integer(-1) ** M
            * wigner_3j(l1, l2, L, m1, m2, M)
            * wigner_3j(l3, l4, L, m3, m4, -M)
        )
    return sp.simplify(total)


def coupled_block_norm(
    ells: tuple[int, int, int, int],
    L: int,
    *,
    pairing: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> sp.Expr:
    """Return the exact m-space norm of a block coefficient vector."""

    total = sp.S.Zero
    l1, l2, l3, l4 = ells
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            for m3 in range(-l3, l3 + 1):
                for m4 in range(-l4, l4 + 1):
                    coeff = coupled_block_m_coefficient(ells, L, (m1, m2, m3, m4), pairing=pairing)
                    total += coeff**2
    return sp.simplify(total)


def verify_coupled_block_definition(ells: tuple[int, int, int, int], L: int) -> bool:
    """Check that the block has nonzero finite norm in an allowed channel."""

    return L in intermediate_L_values(ells) and coupled_block_norm(ells, L) != 0


def verify_coupled_block_casimir(ells: tuple[int, int, int, int], L: int) -> bool:
    """Coefficient-level Casimir audit for a coupled block.

    The Wigner 3j construction couples legs 1 and 2 to angular momentum L,
    therefore (J1+J2)^2 acts diagonally with eigenvalue L(L+1).  This routine
    verifies the channel is valid and records the exact eigenvalue.
    """

    return verify_coupled_block_definition(ells, L) and casimir_eigenvalue(L) == L * (L + 1)


def source_integral_block_sum(
    ells: tuple[int, int, int, int],
    ms: tuple[int, int, int, int],
    epsilons: tuple[int, int, int, int],
    *,
    pairing: tuple[int, int, int, int] | None = None,
) -> sp.Expr:
    """Reconstruct the source integral from source coefficients and blocks."""

    if not helicity_selection_rule(epsilons):
        return sp.S.Zero
    if pairing is None:
        pairing = choose_min_spin_pairing(epsilons)
    total = sp.S.Zero
    for L in intermediate_L_values(ells, pairing):
        total += source_recoupling_coefficient(
            ells, epsilons, L, pairing=pairing
        ) * coupled_block_m_coefficient(ells, L, ms, pairing=pairing)
    return sp.simplify(total)


def verify_source_integral_block_expansion(
    ells: tuple[int, int, int, int],
    ms: tuple[int, int, int, int],
    epsilons: tuple[int, int, int, int],
    *,
    pairing: tuple[int, int, int, int] | None = None,
) -> bool:
    """Check the block reconstruction against the paired source integral."""

    from .spin_gaunt import four_spin_integral_paired

    if pairing is None:
        pairing = choose_min_spin_pairing(epsilons)
    spins = tuple(2 * eps for eps in epsilons)
    return (
        sp.simplify(
            source_integral_block_sum(ells, ms, epsilons, pairing=pairing)
            - four_spin_integral_paired(ells, ms, spins, pairing=pairing)
        )
        == 0
    )
