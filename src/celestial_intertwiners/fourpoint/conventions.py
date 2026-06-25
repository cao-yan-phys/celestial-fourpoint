"""Four-point observable, helicity, and temporal conventions."""

from __future__ import annotations

import itertools
import math

import sympy as sp


ANGULAR_OBSERVABLES = {"P", "A"}
TEMPORAL_OBSERVABLES = {"Z", "P", "A", "ADOT"}


def normalize_observable(observable: str) -> str:
    """Normalize observable names used in the four-point program."""

    obs = observable.upper()
    aliases = {
        "PTA": "P",
        "TIMING": "P",
        "TIMING_RESIDUAL": "P",
        "ASTROMETRY": "A",
        "ASTROMETRIC": "A",
        "PROPER_MOTION": "ADOT",
        "PM": "ADOT",
    }
    return aliases.get(obs, obs)


def validate_epsilon(epsilon: int) -> int:
    """Return epsilon after validating epsilon = +/-1."""

    if epsilon not in {-1, 1}:
        raise ValueError("epsilon must be +1 or -1")
    return epsilon


def tensor_helicity(epsilon: int) -> int:
    """Return physical tensor helicity 2 epsilon."""

    return 2 * validate_epsilon(epsilon)


def source_spin(epsilon: int) -> int:
    """Spin weight carried by the conjugated source harmonic."""

    return 2 * validate_epsilon(epsilon)


def angular_output_spin(observable: str, epsilon: int) -> int:
    """Return external spin weight for a PTA or astrometric angular response."""

    obs = normalize_observable(observable)
    validate_epsilon(epsilon)
    if obs == "P":
        return 0
    if obs == "A":
        return -epsilon
    raise ValueError("angular observable must be P or A")


def helicity_selection_rule(epsilons: tuple[int, int, int, int]) -> bool:
    """Return True when polarization-angle averaging allows the assignment."""

    if len(epsilons) != 4:
        raise ValueError("four-point helicity assignments need four epsilons")
    return all(eps in {-1, 1} for eps in epsilons) and sum(epsilons) == 0


def all_allowed_helicity_assignments() -> tuple[tuple[int, int, int, int], ...]:
    """Return the six two-plus/two-minus helicity assignments."""

    return tuple(
        eps
        for eps in itertools.product((-1, 1), repeat=4)
        if helicity_selection_rule(eps)
    )


def temporal_transfer(observable: str, frequency, *, exact: bool = False):
    """Return the temporal/frequency factor, separate from angular response."""

    obs = normalize_observable(observable)
    if obs not in TEMPORAL_OBSERVABLES:
        raise ValueError("unknown temporal observable")
    if exact:
        f = sp.sympify(frequency)
        if obs in {"Z", "A"}:
            return sp.S.One
        if obs == "P":
            return 1 / (2 * sp.pi * sp.I * f)
        if obs == "ADOT":
            return 2 * sp.pi * sp.I * f
    else:
        if obs in {"Z", "A"}:
            return 1.0
        if obs == "P":
            return 1 / (2j * math.pi * frequency)
        if obs == "ADOT":
            return 2j * math.pi * frequency
    raise ValueError("unreachable observable branch")
