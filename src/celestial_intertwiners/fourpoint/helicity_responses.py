"""Coefficient-level helicity response expansions."""

from __future__ import annotations

import sympy as sp

from ..transfers import (
    astrometry_transfer,
    astrometry_transfer_exact,
    pta_transfer,
    pta_transfer_exact,
)
from .conventions import angular_output_spin, normalize_observable, validate_epsilon


def response_transfer(observable: str, ell: int, *, exact: bool = False):
    """Return angular response transfer r_l^P or r_l^A."""

    obs = normalize_observable(observable)
    if obs == "P":
        return pta_transfer_exact(ell) if exact else pta_transfer(ell)
    if obs == "A":
        return astrometry_transfer_exact(ell) if exact else astrometry_transfer(ell)
    raise ValueError("angular observable must be P or A")


def response_spins(observable: str, epsilon: int) -> tuple[int, int]:
    """Return (external spin, source spin) for the bi-spin response term."""

    validate_epsilon(epsilon)
    return angular_output_spin(observable, epsilon), 2 * epsilon


def descendant_operator_factor(epsilon: int, ell: int, *, exact: bool = False):
    """Return the factor from D^epsilon A^epsilon to P^epsilon.

    With Goldberg signs, D^+ = eth on spin -1 and D^- = -bar_eth on spin +1,
    so both channels contribute sqrt(l(l+1)).
    """

    validate_epsilon(epsilon)
    if ell < 1:
        raise ValueError("ell must be positive")
    return sp.sqrt(ell * (ell + 1)) if exact else (ell * (ell + 1)) ** 0.5


def verify_transfer_normalization(ell: int) -> bool:
    """Check D^epsilon r_l^A = r_l^P for both epsilon signs."""

    lhs = astrometry_transfer_exact(ell) * sp.sqrt(ell * (ell + 1))
    rhs = pta_transfer_exact(ell)
    return sp.simplify(lhs - rhs) == 0


def bispin_expansion_metadata(observable: str, epsilon: int, ell: int) -> dict[str, object]:
    """Return the coefficient and spin data for one response expansion term."""

    return {
        "observable": normalize_observable(observable),
        "epsilon": validate_epsilon(epsilon),
        "ell": ell,
        "transfer": response_transfer(observable, ell, exact=True),
        "external_spin": angular_output_spin(observable, epsilon),
        "source_spin": 2 * epsilon,
    }
