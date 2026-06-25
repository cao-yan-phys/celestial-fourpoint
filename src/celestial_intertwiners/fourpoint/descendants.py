"""Spin-descendant factors for mixed four-point kernels."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from .conventions import normalize_observable, validate_epsilon


def descendant_factor(epsilon: int, ell: int, *, exact: bool = False):
    """Return D^epsilon acting on an astrometric external leg."""

    validate_epsilon(epsilon)
    if ell < 1:
        raise ValueError("ell must be positive")
    return sp.sqrt(ell * (ell + 1)) if exact else (ell * (ell + 1)) ** 0.5


def inverse_descendant_factor(epsilon: int, ell: int, *, exact: bool = False):
    """Return the inverse descendant on the physical ell >= 2 sector."""

    if ell < 2:
        raise ValueError("inverse descendant is defined on ell >= 2 response modes")
    factor = descendant_factor(epsilon, ell, exact=exact)
    return 1 / factor


def mixed_descendant_to_pppp_factor(
    observables: Sequence[str],
    epsilons: Sequence[int],
    ells: Sequence[int],
    *,
    exact: bool = False,
):
    """Return product of D factors mapping mixed A/P kernel to PPPP."""

    if not (len(observables) == len(epsilons) == len(ells) == 4):
        raise ValueError("four observables, epsilons, and ells are required")
    factor = sp.S.One if exact else 1.0
    for observable, epsilon, ell in zip(observables, epsilons, ells):
        if normalize_observable(observable) == "A":
            factor *= descendant_factor(epsilon, ell, exact=exact)
    return sp.simplify(factor) if exact else factor


def pppp_to_mixed_inverse_factor(
    observables: Sequence[str],
    epsilons: Sequence[int],
    ells: Sequence[int],
    *,
    exact: bool = False,
):
    """Return product of inverse descendants generating a mixed kernel from PPPP."""

    if not (len(observables) == len(epsilons) == len(ells) == 4):
        raise ValueError("four observables, epsilons, and ells are required")
    factor = sp.S.One if exact else 1.0
    for observable, epsilon, ell in zip(observables, epsilons, ells):
        if normalize_observable(observable) == "A":
            factor *= inverse_descendant_factor(epsilon, ell, exact=exact)
    return sp.simplify(factor) if exact else factor
