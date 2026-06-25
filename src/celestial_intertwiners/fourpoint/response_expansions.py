"""Bi-spin response expansion metadata and coefficient audits."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .conventions import angular_output_spin, source_spin
from .helicity_responses import descendant_operator_factor, response_transfer


@dataclass(frozen=True)
class BispinResponseTerm:
    """One harmonic term in a PTA or astrometric bi-spin expansion."""

    observable: str
    epsilon: int
    ell: int
    external_spin: int
    source_spin: int
    transfer: sp.Expr


def bispin_response_term(observable: str, epsilon: int, ell: int) -> BispinResponseTerm:
    """Return the coefficient/spin metadata for one response term."""

    return BispinResponseTerm(
        observable=observable.upper(),
        epsilon=epsilon,
        ell=ell,
        external_spin=angular_output_spin(observable, epsilon),
        source_spin=source_spin(epsilon),
        transfer=response_transfer(observable, ell, exact=True),
    )


def descendant_coefficient_residual(epsilon: int, ell: int) -> sp.Expr:
    """Return D^epsilon A_l - P_l at coefficient level."""

    astrometry = response_transfer("A", ell, exact=True)
    pta = response_transfer("P", ell, exact=True)
    return sp.simplify(descendant_operator_factor(epsilon, ell, exact=True) * astrometry - pta)


def verify_bispin_descendant_coefficients(lmax: int = 12) -> bool:
    """Verify coefficient-level descendant identity for ell=2..lmax."""

    return all(
        descendant_coefficient_residual(epsilon, ell) == 0
        for epsilon in (-1, 1)
        for ell in range(2, lmax + 1)
    )
