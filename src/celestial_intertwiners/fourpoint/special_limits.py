"""Pair-coincident and all-equal four-point special-limit audits."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .block_benchmark import (
    benchmark_mixed_descendant_vs_direct,
    benchmark_pppp_block_vs_direct,
)
from .kuntz_benchmark import load_kuntz_fixtures


@dataclass(frozen=True)
class SpecialLimitCase:
    """One degenerate four-direction configuration."""

    name: str
    vectors: np.ndarray


@dataclass(frozen=True)
class SpecialLimitObservableResult:
    """One observable check in a special-limit configuration."""

    case_name: str
    observables: tuple[str, str, str, str]
    direct: complex
    reconstructed: complex
    absolute_error: float
    relative_error: float
    passed: bool


@dataclass(frozen=True)
class SpecialLimitAuditResult:
    """Collection of special-limit checks."""

    results: tuple[SpecialLimitObservableResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def max_relative_error(self) -> float:
        if not self.results:
            return 0.0
        return max(result.relative_error for result in self.results)

    @property
    def max_absolute_error(self) -> float:
        if not self.results:
            return 0.0
        return max(result.absolute_error for result in self.results)


def special_limit_cases() -> tuple[SpecialLimitCase, ...]:
    """Return deterministic pair-coincident and all-equal Kuntz-seed limits."""

    base = load_kuntz_fixtures()[0].vectors
    p1, p2, p3, p4 = base
    return (
        SpecialLimitCase("a_equals_d", np.asarray([p1, p2, p3, p1])),
        SpecialLimitCase("a_equals_c", np.asarray([p1, p2, p1, p4])),
        SpecialLimitCase("a_equals_b", np.asarray([p1, p1, p3, p4])),
        SpecialLimitCase("all_equal", np.asarray([p1, p1, p1, p1])),
    )


def _passes(absolute_error: float, relative_error: float, tolerance: float, absolute_tolerance: float) -> bool:
    return absolute_error < absolute_tolerance or relative_error < tolerance


def audit_special_limit_case(
    case: SpecialLimitCase,
    *,
    lmax: int = 4,
    tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
    n_theta: int = 10,
    n_phi: int = 20,
) -> tuple[SpecialLimitObservableResult, ...]:
    """Check PPPP/APPP/AAPP in one degenerate configuration."""

    pppp = benchmark_pppp_block_vs_direct(
        vectors=case.vectors,
        lmax=lmax,
        tolerance=tolerance,
        n_theta=n_theta,
        n_phi=n_phi,
    )
    observable_results = [
        SpecialLimitObservableResult(
            case_name=case.name,
            observables=("P", "P", "P", "P"),
            direct=pppp.direct,
            reconstructed=pppp.block,
            absolute_error=float(abs(pppp.block - pppp.direct)),
            relative_error=pppp.relative_error,
            passed=_passes(
                float(abs(pppp.block - pppp.direct)),
                pppp.relative_error,
                tolerance,
                absolute_tolerance,
            ),
        )
    ]
    for observables in (("A", "P", "P", "P"), ("A", "A", "P", "P")):
        mixed = benchmark_mixed_descendant_vs_direct(
            observables,
            vectors=case.vectors,
            lmax=lmax,
            tolerance=tolerance,
            n_theta=n_theta,
            n_phi=n_phi,
        )
        absolute_error = float(abs(mixed.descendant - mixed.direct))
        observable_results.append(
            SpecialLimitObservableResult(
                case_name=case.name,
                observables=mixed.observables,
                direct=mixed.direct,
                reconstructed=mixed.descendant,
                absolute_error=absolute_error,
                relative_error=mixed.relative_error,
                passed=_passes(
                    absolute_error,
                    mixed.relative_error,
                    tolerance,
                    absolute_tolerance,
                ),
            )
        )
    return tuple(observable_results)


@lru_cache(maxsize=8)
def audit_special_limits(
    *,
    lmax: int = 4,
    tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
    n_theta: int = 10,
    n_phi: int = 20,
) -> SpecialLimitAuditResult:
    """Run pair-coincident and all-equal finite-kernel audits."""

    results: list[SpecialLimitObservableResult] = []
    for case in special_limit_cases():
        results.extend(
            audit_special_limit_case(
                case,
                lmax=lmax,
                tolerance=tolerance,
                absolute_tolerance=absolute_tolerance,
                n_theta=n_theta,
                n_phi=n_phi,
            )
        )
    return SpecialLimitAuditResult(results=tuple(results))


def verify_pair_coincident_special_limits(
    *,
    lmax: int = 4,
    tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> bool:
    """Return True when all pair-coincident finite-kernel checks pass."""

    audit = audit_special_limits(
        lmax=lmax,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    return all(result.passed for result in audit.results if result.case_name != "all_equal")


def verify_all_equal_special_limit(
    *,
    lmax: int = 4,
    tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> bool:
    """Return True when the all-equal finite-kernel checks pass."""

    audit = audit_special_limits(
        lmax=lmax,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    return all(result.passed for result in audit.results if result.case_name == "all_equal")
