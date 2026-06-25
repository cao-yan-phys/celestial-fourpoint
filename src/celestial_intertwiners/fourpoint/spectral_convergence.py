"""Physical spectral convergence checks against the Kuntz closed-form seed."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

from ..geometry import spherical_angles_from_direction
from .conventions import angular_output_spin, normalize_observable, validate_epsilon
from .direct_quadrature import spin_weighted_spherical_harmonic
from .helicity_responses import response_transfer
from .kuntz_benchmark import KUNTZ_SEED_EPSILONS, load_kuntz_fixtures
from .pointwise_descendant import explicit_astrometry_antenna, explicit_pta_antenna

PHYSICAL_PTA_TRANSFER_SCALE = 2.0 * math.pi


@dataclass(frozen=True)
class SpectralKuntzEntry:
    """One physical finite-spectral value compared with the Kuntz seed."""

    lmax: int
    value: complex
    relative_error: float


@dataclass(frozen=True)
class SpectralKuntzConvergenceResult:
    """Raw and tail-averaged spectral convergence against a Kuntz fixture."""

    fixture_name: str
    reference: complex
    entries: tuple[SpectralKuntzEntry, ...]
    tail_average: complex
    tail_relative_error: float
    passed: bool


@dataclass(frozen=True)
class MixedPhysicalComparisonResult:
    """Physical explicit mixed integral vs. spectral descendant value."""

    observables: tuple[str, str, str, str]
    lmax: int
    direct: complex
    spectral: complex
    relative_error: float
    passed: bool


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("direction must be nonzero")
    return vector / norm


def _validate_observables(observables) -> tuple[str, str, str, str]:
    if len(observables) != 4:
        raise ValueError("four observables are required")
    normalized = tuple(normalize_observable(observable) for observable in observables)
    if any(observable not in {"P", "A"} for observable in normalized):
        raise ValueError("spectral convergence supports only P and A observables")
    return normalized


def _validate_epsilons(epsilons) -> tuple[int, int, int, int]:
    if len(epsilons) != 4:
        raise ValueError("four helicities are required")
    return tuple(validate_epsilon(epsilon) for epsilon in epsilons)


@lru_cache(maxsize=16)
def _source_grid(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, theta_weights = leggauss(n_theta)
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    thetas = np.arccos(xs)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing="ij")
    weights = np.repeat(theta_weights, n_phi) * (2.0 * math.pi / n_phi)
    return theta_grid.ravel(), phi_grid.ravel(), weights


@lru_cache(maxsize=64)
def _source_harmonic_matrix(
    lmax: int,
    spin: int,
    n_theta: int,
    n_phi: int,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    theta_grid, phi_grid, _weights = _source_grid(n_theta, n_phi)
    modes: list[tuple[int, int]] = []
    rows: list[np.ndarray] = []
    for ell in range(2, lmax + 1):
        for m in range(-ell, ell + 1):
            modes.append((ell, m))
            rows.append(
                np.asarray(
                    [
                        spin_weighted_spherical_harmonic(ell, m, spin, float(theta), float(phi))
                        for theta, phi in zip(theta_grid, phi_grid)
                    ],
                    dtype=complex,
                )
            )
    return tuple(modes), np.vstack(rows)


def _external_coefficients(
    direction,
    observable: str,
    epsilon: int,
    modes: tuple[tuple[int, int], ...],
    *,
    transfer_scale: float,
) -> np.ndarray:
    theta, phi = spherical_angles_from_direction(_unit(direction))
    external_spin = angular_output_spin(observable, epsilon)
    coefficients = np.empty(len(modes), dtype=complex)
    for index, (ell, m) in enumerate(modes):
        coefficients[index] = (
            transfer_scale
            * response_transfer(observable, ell, exact=False)
            * spin_weighted_spherical_harmonic(ell, m, external_spin, theta, phi)
        )
    return coefficients


def spectral_truncated_direct_average(
    observables,
    vectors,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    lmax: int,
    n_theta: int | None = None,
    n_phi: int | None = None,
    transfer_scale: float = PHYSICAL_PTA_TRANSFER_SCALE,
) -> complex:
    """Evaluate a finite physical spectral four-point average with cached harmonics."""

    if lmax < 2:
        raise ValueError("lmax must be at least 2")
    observables = _validate_observables(observables)
    epsilons = _validate_epsilons(epsilons)
    unit_vectors = tuple(_unit(vector) for vector in vectors)
    if len(unit_vectors) != 4:
        raise ValueError("four directions are required")
    n_theta = 2 * lmax + 8 if n_theta is None else n_theta
    n_phi = 4 * lmax + 16 if n_phi is None else n_phi
    _theta_grid, _phi_grid, weights = _source_grid(n_theta, n_phi)

    product = np.ones(weights.shape, dtype=complex)
    for observable, vector, epsilon in zip(observables, unit_vectors, epsilons):
        source_spin = 2 * epsilon
        modes, source_values = _source_harmonic_matrix(lmax, source_spin, n_theta, n_phi)
        coefficients = _external_coefficients(
            vector,
            observable,
            epsilon,
            modes,
            transfer_scale=transfer_scale,
        )
        product *= coefficients @ source_values
    return complex(np.dot(weights, product) / (4.0 * math.pi))


def pppp_physical_spectral_average(
    vectors,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    lmax: int,
    n_theta: int | None = None,
    n_phi: int | None = None,
) -> complex:
    """Evaluate the physically normalized finite PPPP spectral average."""

    return spectral_truncated_direct_average(
        ("P", "P", "P", "P"),
        vectors,
        epsilons,
        lmax=lmax,
        n_theta=n_theta,
        n_phi=n_phi,
    )


def _explicit_physical_antenna(observable: str, vector, omega, epsilon: int) -> complex:
    theta, phi = spherical_angles_from_direction(_unit(vector))
    if observable == "P":
        return explicit_pta_antenna(theta, phi, omega, epsilon)
    if observable == "A":
        # The spectral mixed-kernel convention carries an extra helicity sign
        # relative to the normalized pointwise descendant component.
        return epsilon * explicit_astrometry_antenna(theta, phi, omega, epsilon)
    raise ValueError("observable must be P or A")


def explicit_physical_mixed_average(
    observables,
    vectors,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    n_theta: int = 72,
    n_phi: int = 144,
) -> complex:
    """Directly integrate explicit physical P/A helicity antennas."""

    observables = _validate_observables(observables)
    epsilons = _validate_epsilons(epsilons)
    unit_vectors = tuple(_unit(vector) for vector in vectors)
    if len(unit_vectors) != 4:
        raise ValueError("four directions are required")
    theta_grid, phi_grid, weights = _source_grid(n_theta, n_phi)
    total = 0.0j
    for theta, phi, weight in zip(theta_grid, phi_grid, weights):
        sin_theta = math.sin(float(theta))
        omega = np.array(
            [
                sin_theta * math.cos(float(phi)),
                sin_theta * math.sin(float(phi)),
                math.cos(float(theta)),
            ]
        )
        value = 1.0 + 0.0j
        for observable, vector, epsilon in zip(observables, unit_vectors, epsilons):
            value *= _explicit_physical_antenna(observable, vector, omega, epsilon)
        total += weight * value
    return complex(total / (4.0 * math.pi))


def compare_physical_mixed_spectral(
    observables,
    vectors,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    lmax: int = 12,
    tolerance: float = 2.5e-2,
    direct_n_theta: int = 72,
    direct_n_phi: int = 144,
) -> MixedPhysicalComparisonResult:
    """Compare explicit physical mixed integral to descendant spectral truncation."""

    observables = _validate_observables(observables)
    direct = explicit_physical_mixed_average(
        observables,
        vectors,
        epsilons,
        n_theta=direct_n_theta,
        n_phi=direct_n_phi,
    )
    spectral = spectral_truncated_direct_average(
        observables,
        vectors,
        epsilons,
        lmax=lmax,
    )
    relative = abs(spectral - direct) / max(abs(direct), abs(spectral), 1e-30)
    return MixedPhysicalComparisonResult(
        observables=observables,
        lmax=lmax,
        direct=direct,
        spectral=spectral,
        relative_error=float(relative),
        passed=bool(relative < tolerance),
    )


def benchmark_pppp_spectral_convergence_to_kuntz(
    *,
    fixture_index: int = 0,
    lmax_values: tuple[int, ...] = (6, 12, 16, 18),
    tail_count: int = 3,
    tolerance: float = 2.0e-3,
) -> SpectralKuntzConvergenceResult:
    """Compare physical finite spectral sums with the Kuntz closed-form seed."""

    if tail_count < 1:
        raise ValueError("tail_count must be positive")
    fixture = load_kuntz_fixtures()[fixture_index]
    entries: list[SpectralKuntzEntry] = []
    for lmax in lmax_values:
        value = pppp_physical_spectral_average(fixture.vectors, KUNTZ_SEED_EPSILONS, lmax=lmax)
        relative = abs(value - fixture.value) / max(abs(fixture.value), 1e-30)
        entries.append(SpectralKuntzEntry(lmax=lmax, value=value, relative_error=float(relative)))
    tail_values = [entry.value for entry in entries[-tail_count:]]
    tail_average = complex(sum(tail_values) / len(tail_values))
    tail_relative = abs(tail_average - fixture.value) / max(abs(fixture.value), 1e-30)
    return SpectralKuntzConvergenceResult(
        fixture_name=fixture.name,
        reference=fixture.value,
        entries=tuple(entries),
        tail_average=tail_average,
        tail_relative_error=float(tail_relative),
        passed=bool(tail_relative < tolerance),
    )


@lru_cache(maxsize=4)
def verify_pppp_spectral_convergence_to_kuntz(
    *,
    tolerance: float = 2.0e-3,
) -> bool:
    """Return True when the tail-averaged physical spectral sum matches Kuntz."""

    return benchmark_pppp_spectral_convergence_to_kuntz(tolerance=tolerance).passed
