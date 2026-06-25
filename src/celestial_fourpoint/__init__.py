"""Public calculator API for celestial four-point kernels."""

from celestial_intertwiners.fourpoint import build_precomputed_spectral_kernel
from celestial_intertwiners.fourpoint.kuntz_benchmark import (
    KUNTZ_SEED_EPSILONS,
    load_kuntz_fixtures,
)

from .accuracy import FixtureAccuracyReport, fixture_accuracy_sweep
from .calculator import FourPointCalculator

__all__ = [
    "FixtureAccuracyReport",
    "FourPointCalculator",
    "KUNTZ_SEED_EPSILONS",
    "build_precomputed_spectral_kernel",
    "fixture_accuracy_sweep",
    "load_kuntz_fixtures",
]
