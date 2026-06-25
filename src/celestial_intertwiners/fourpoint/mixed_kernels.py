"""Mixed APPP/AAPP kernel generation scaffolding."""

from __future__ import annotations

from .descendants import pppp_to_mixed_inverse_factor


def mixed_from_pppp_factor(observables, epsilons, ells, *, exact=True):
    """Return the inverse-descendant factor generating a mixed kernel from PPPP."""

    return pppp_to_mixed_inverse_factor(observables, epsilons, ells, exact=exact)
