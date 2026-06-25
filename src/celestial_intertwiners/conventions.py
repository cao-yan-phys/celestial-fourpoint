"""Shared conventions for celestial intertwiners.

The package uses Goldberg spin-weighted harmonic signs:

eth_sY_lm = +sqrt((l-s)(l+s+1)) s+1Y_lm
bar_eth_sY_lm = -sqrt((l+s)(l-s+1)) s-1Y_lm
"""

from __future__ import annotations

from enum import IntEnum


class TensorHelicity(IntEnum):
    """Tensor-GR helicities."""

    PLUS = 2
    MINUS = -2


PTA_SPIN = 0
ASTROMETRY_SPIN_FOR_PLUS_HELICITY = -1
ASTROMETRY_SPIN_FOR_MINUS_HELICITY = 1
MINIMUM_TENSOR_L = 2

GOLDBERG_ETH_SIGN = 1
GOLDBERG_BAR_ETH_SIGN = -1


def astrometry_spin_from_helicity(helicity: int) -> int:
    """Map tensor GW helicity to astrometric spin weight."""

    if helicity == 2:
        return ASTROMETRY_SPIN_FOR_PLUS_HELICITY
    if helicity == -2:
        return ASTROMETRY_SPIN_FOR_MINUS_HELICITY
    raise ValueError("tensor helicity must be +2 or -2")
