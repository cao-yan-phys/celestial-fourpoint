"""External observable legs."""

from __future__ import annotations

from dataclasses import dataclass

from .conventions import astrometry_spin_from_helicity
from .transfers import astrometry_transfer, astrometry_transfer_exact, pta_transfer, pta_transfer_exact


@dataclass(frozen=True)
class Leg:
    """One tensor-source external leg observed by PTA or astrometry."""

    observable: str
    helicity: int

    def __post_init__(self):
        object.__setattr__(self, "observable", self.observable.upper())
        if self.observable not in {"PTA", "ASTROMETRY"}:
            raise ValueError("observable must be 'PTA' or 'ASTROMETRY'")
        if self.helicity not in {2, -2}:
            raise ValueError("tensor helicity must be +2 or -2")


def leg_transfer(leg: Leg, l: int, *, exact: bool = False):
    """Return the transfer factor for one external leg."""

    if leg.observable == "PTA":
        return pta_transfer_exact(l) if exact else pta_transfer(l)
    if leg.observable == "ASTROMETRY":
        return astrometry_transfer_exact(l) if exact else astrometry_transfer(l)
    raise ValueError(f"unknown observable {leg.observable!r}")


def output_spin(leg: Leg) -> int:
    """Return output spin weight for an observable leg."""

    if leg.observable == "PTA":
        return 0
    if leg.observable == "ASTROMETRY":
        return astrometry_spin_from_helicity(leg.helicity)
    raise ValueError(f"unknown observable {leg.observable!r}")
