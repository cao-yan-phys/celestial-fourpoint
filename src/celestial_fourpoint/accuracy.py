"""Accuracy helpers for the public four-point calculator."""

from __future__ import annotations

from dataclasses import dataclass

from celestial_intertwiners.fourpoint.kuntz_benchmark import (
    KUNTZ_SEED_EPSILONS,
    load_kuntz_fixtures,
)

from .calculator import FourPointCalculator


@dataclass(frozen=True)
class FixtureAccuracyReport:
    """Accuracy summary for one Kuntz fixture."""

    fixture_name: str
    pppp_kuntz_relative_error: float
    max_cached_direct_relative_error: float


def fixture_accuracy_sweep(
    *,
    lmax: int = 15,
    fixture_count: int | None = None,
    direct_n_theta: int = 160,
    direct_n_phi: int = 320,
    backend: str = "auto",
) -> tuple[FixtureAccuracyReport, ...]:
    """Compare cached kernels with direct quadrature on Kuntz fixtures."""

    calculator = FourPointCalculator.load_precomputed(lmax=lmax)
    fixtures = load_kuntz_fixtures()
    if fixture_count is not None:
        fixtures = fixtures[:fixture_count]

    reports = []
    for fixture in fixtures:
        pppp_cached = calculator.evaluate(("P", "P", "P", "P"), fixture.vectors, KUNTZ_SEED_EPSILONS)
        pppp_kuntz = calculator.kuntz_pppp(fixture.vectors)
        pppp_error = abs(pppp_cached - pppp_kuntz) / max(abs(pppp_kuntz), 1e-30)
        entries = calculator.compare(
            fixture.vectors,
            KUNTZ_SEED_EPSILONS,
            n_theta=direct_n_theta,
            n_phi=direct_n_phi,
            backend=backend,
        )
        reports.append(
            FixtureAccuracyReport(
                fixture_name=fixture.name,
                pppp_kuntz_relative_error=float(pppp_error),
                max_cached_direct_relative_error=max(entry.relative_error for entry in entries),
            )
        )
    return tuple(reports)
