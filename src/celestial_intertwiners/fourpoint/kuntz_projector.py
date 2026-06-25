"""External-leg projectors built on the Kuntz PPPP closed-form master.

The Kuntz-master projector is reliable for one astrometric external leg and is
currently experimental for multiple astrometric legs.  Multi-leg projection
with the simple Sobol product-sphere estimator is variance limited; use the
spectral convergence helpers in this module for production AAPP/AAAP/AAAA
checks until a deterministic external-projector contraction is implemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.stats import qmc

from ..geometry import spherical_angles_from_direction
from .conventions import angular_output_spin, normalize_observable, validate_epsilon
from .direct_quadrature import sph_harm_y, spin_weighted_spherical_harmonic
from .fast_evaluator import FastFourPointEvaluator
from .kuntz_benchmark import KUNTZ_SEED_EPSILONS, load_kuntz_fixtures
from .kuntz_formula import _cached_lambdified, _path_key
from .mixed_closed_form import kuntz_canonical_vectors_from_angles
from .spectral_sum_evaluator import SpectralSumEvaluator


@dataclass(frozen=True)
class KuntzProjectorConvergenceEntry:
    """One finite external-projector comparison against direct quadrature."""

    observables: tuple[str, str, str, str]
    lmax: int
    value: complex
    direct: complex
    relative_error: float
    finite_fraction: float
    sample_count: int


@dataclass(frozen=True)
class KuntzProjectorConvergenceResult:
    """Convergence sweep for one mixed kernel."""

    observables: tuple[str, str, str, str]
    entries: tuple[KuntzProjectorConvergenceEntry, ...]

    @property
    def best_relative_error(self) -> float:
        return min((entry.relative_error for entry in self.entries), default=math.inf)


@dataclass(frozen=True)
class MixedSpectralConvergenceEntry:
    """One reliable full-spectral mixed-kernel comparison."""

    observables: tuple[str, str, str, str]
    lmax: int
    value: complex
    direct: complex
    relative_error: float


@dataclass(frozen=True)
class MixedSpectralConvergenceResult:
    """Reliable full-spectral convergence sweep for one mixed kernel."""

    observables: tuple[str, str, str, str]
    entries: tuple[MixedSpectralConvergenceEntry, ...]

    @property
    def best_relative_error(self) -> float:
        return min((entry.relative_error for entry in self.entries), default=math.inf)


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("direction vectors must be nonzero")
    return vectors / norms


def _validate_observables(observables) -> tuple[str, str, str, str]:
    if len(observables) != 4:
        raise ValueError("four observables are required")
    normalized = tuple(normalize_observable(observable) for observable in observables)
    if any(observable not in {"P", "A"} for observable in normalized):
        raise ValueError("Kuntz projector supports only P/A observables")
    return normalized


def _validate_epsilons(epsilons) -> tuple[int, int, int, int]:
    if len(epsilons) != 4:
        raise ValueError("four helicities are required")
    return tuple(validate_epsilon(epsilon) for epsilon in epsilons)


def _sobol_sphere_samples(*, dimensions: int, sample_power: int, seed: int) -> np.ndarray:
    if dimensions <= 0:
        return np.empty((2**sample_power, 0, 3), dtype=float)
    sampler = qmc.Sobol(d=2 * dimensions, scramble=True, seed=seed)
    unit = sampler.random_base2(sample_power)
    samples = np.empty((unit.shape[0], dimensions, 3), dtype=float)
    for index in range(dimensions):
        z = 2.0 * unit[:, 2 * index] - 1.0
        phi = 2.0 * math.pi * unit[:, 2 * index + 1]
        radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        samples[:, index, 0] = radius * np.cos(phi)
        samples[:, index, 1] = radius * np.sin(phi)
        samples[:, index, 2] = z
    return samples


def _angles_from_vectors_batch(vectors: np.ndarray) -> tuple[np.ndarray, ...]:
    unit = _unit_rows(np.asarray(vectors, dtype=float))
    p1 = unit[:, 0, :]
    p2 = unit[:, 1, :]
    p3 = unit[:, 2, :]
    p4 = unit[:, 3, :]
    c12 = np.einsum("ij,ij->i", p1, p2)
    c13 = np.einsum("ij,ij->i", p1, p3)
    c14 = np.einsum("ij,ij->i", p1, p4)
    normal = np.cross(p1, p2)

    def psi_for(point: np.ndarray, cosine_to_p1: np.ndarray) -> np.ndarray:
        denominator = np.sqrt(np.maximum(0.0, 1.0 - c12 * c12)) * np.sqrt(
            np.maximum(0.0, 1.0 - cosine_to_p1 * cosine_to_p1)
        )
        numerator = np.einsum("ij,ij->i", p2, point) - c12 * cosine_to_p1
        cosine = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 1e-14,
        )
        sign = np.sign(np.einsum("ij,ij->i", point, normal))
        sign = np.where(sign == 0.0, 1.0, sign)
        return sign * np.arccos(np.clip(cosine, -1.0, 1.0))

    return c12, c13, c14, psi_for(p3, c13), psi_for(p4, c14)


@lru_cache(maxsize=64)
def _target_spin_harmonics(
    lmax: int,
    spin: int,
    theta: float,
    phi: float,
) -> tuple[tuple[int, int, complex], ...]:
    return tuple(
        (
            ell,
            mode,
            spin_weighted_spherical_harmonic(ell, mode, spin, theta, phi)
            / math.sqrt(ell * (ell + 1)),
        )
        for ell in range(2, lmax + 1)
        for mode in range(-ell, ell + 1)
    )


def _external_projector_kernel(
    sample_vectors: np.ndarray,
    target_vector: np.ndarray,
    *,
    spin: int,
    lmax: int,
) -> np.ndarray:
    target_theta, target_phi = spherical_angles_from_direction(target_vector)
    sample_theta = np.arccos(np.clip(sample_vectors[:, 2], -1.0, 1.0))
    sample_phi = np.arctan2(sample_vectors[:, 1], sample_vectors[:, 0])
    kernel = np.zeros(sample_vectors.shape[0], dtype=complex)
    for ell, mode, target_value in _target_spin_harmonics(
        lmax,
        spin,
        float(target_theta),
        float(target_phi),
    ):
        kernel += target_value * np.conj(sph_harm_y(ell, mode, sample_theta, sample_phi))
    return 4.0 * math.pi * kernel


def evaluate_mixed_projector_from_kuntz_master_angles(
    observables,
    angles,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    lmax: int,
    sample_power: int = 13,
    seed: int = 20260625,
) -> tuple[complex, float]:
    """Evaluate a semi-closed mixed kernel by projecting Kuntz PPPP externally.

    The source-direction integral is not recomputed here.  It is already
    contained in the Kuntz PPPP closed form.  The only approximation is the
    finite external-leg projector and the Sobol product-sphere quadrature used
    for multiple astrometric legs.

    Warning: the current Sobol product-sphere implementation is experimental
    for two or more astrometric legs and is not a production AAAP/AAAA route.
    """

    if lmax < 2:
        raise ValueError("lmax must be at least 2")
    observables = _validate_observables(observables)
    epsilons = _validate_epsilons(epsilons)
    vectors = kuntz_canonical_vectors_from_angles(angles)
    astrometric_legs = [index for index, observable in enumerate(observables) if observable == "A"]
    if not astrometric_legs:
        evaluator = _cached_lambdified(_path_key(None))
        return complex(evaluator(*angles)), 1.0

    samples = _sobol_sphere_samples(
        dimensions=len(astrometric_legs),
        sample_power=sample_power,
        seed=seed,
    )
    sample_count = samples.shape[0]
    sampled_vectors = np.broadcast_to(vectors, (sample_count, 4, 3)).copy()
    kernel_product = np.ones(sample_count, dtype=complex)
    for sample_axis, leg_index in enumerate(astrometric_legs):
        sampled_vectors[:, leg_index, :] = samples[:, sample_axis, :]
        kernel_product *= _external_projector_kernel(
            samples[:, sample_axis, :],
            vectors[leg_index],
            spin=angular_output_spin("A", epsilons[leg_index]),
            lmax=lmax,
        )

    cb, cc, cd, psi_c, psi_d = _angles_from_vectors_batch(sampled_vectors)
    evaluator = _cached_lambdified(_path_key(None))
    with np.errstate(all="ignore"):
        master_values = np.asarray(
            evaluator(
                cb.astype(complex),
                cc.astype(complex),
                cd.astype(complex),
                psi_c.astype(complex),
                psi_d.astype(complex),
            ),
            dtype=complex,
        )
    integrand = master_values * kernel_product
    finite = np.isfinite(integrand.real) & np.isfinite(integrand.imag)
    if not np.any(finite):
        return complex(np.nan, np.nan), 0.0
    return complex(np.mean(integrand[finite])), float(np.count_nonzero(finite) / sample_count)


def benchmark_mixed_projector_convergence(
    observables,
    *,
    fixture_index: int = 0,
    lmax_values: tuple[int, ...] = (4, 6, 8, 10),
    sample_power: int = 13,
    seed: int = 20260625,
    direct_n_theta: int = 160,
    direct_n_phi: int = 320,
    backend: str = "auto",
    vary_seed_by_lmax: bool = False,
) -> KuntzProjectorConvergenceResult:
    """Sweep finite external-projector cutoffs against direct antenna quadrature.

    This benchmarks the experimental Kuntz-master projector.  For reliable
    AAPP/AAAP/AAAA production checks, use
    :func:`benchmark_mixed_spectral_convergence`.
    """

    observables = _validate_observables(observables)
    fixture = load_kuntz_fixtures()[fixture_index]
    vectors = kuntz_canonical_vectors_from_angles(fixture.angles)
    direct = FastFourPointEvaluator(
        n_theta=direct_n_theta,
        n_phi=direct_n_phi,
        backend=backend,
    ).evaluate(observables, vectors, KUNTZ_SEED_EPSILONS).value
    entries = []
    for lmax in lmax_values:
        value, finite_fraction = evaluate_mixed_projector_from_kuntz_master_angles(
            observables,
            fixture.angles,
            KUNTZ_SEED_EPSILONS,
            lmax=lmax,
            sample_power=sample_power,
            seed=seed + 1009 * int(lmax) if vary_seed_by_lmax else seed,
        )
        relative_error = abs(value - direct) / max(abs(direct), 1e-30)
        entries.append(
            KuntzProjectorConvergenceEntry(
                observables=observables,
                lmax=int(lmax),
                value=value,
                direct=direct,
                relative_error=float(relative_error),
                finite_fraction=finite_fraction,
                sample_count=2**sample_power,
            )
        )
    return KuntzProjectorConvergenceResult(observables=observables, entries=tuple(entries))


def evaluate_mixed_spectral_kernel(
    observables,
    angles,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    lmax: int,
    backend: str = "auto",
) -> complex:
    """Evaluate the reliable full-spectral mixed kernel in the Kuntz frame."""

    observables = _validate_observables(observables)
    vectors = kuntz_canonical_vectors_from_angles(angles)
    return SpectralSumEvaluator(lmax=lmax, backend=backend).evaluate(
        observables,
        vectors,
        epsilons,
    ).value


def benchmark_mixed_spectral_convergence(
    observables,
    *,
    fixture_index: int = 0,
    lmax_values: tuple[int, ...] = (4, 6, 8, 10, 12),
    direct_n_theta: int = 160,
    direct_n_phi: int = 320,
    backend: str = "auto",
) -> MixedSpectralConvergenceResult:
    """Sweep the reliable full-spectral mixed evaluator against direct quadrature."""

    observables = _validate_observables(observables)
    fixture = load_kuntz_fixtures()[fixture_index]
    vectors = kuntz_canonical_vectors_from_angles(fixture.angles)
    direct = FastFourPointEvaluator(
        n_theta=direct_n_theta,
        n_phi=direct_n_phi,
        backend=backend,
    ).evaluate(observables, vectors, KUNTZ_SEED_EPSILONS).value
    entries = []
    for lmax in lmax_values:
        value = SpectralSumEvaluator(lmax=int(lmax), backend=backend).evaluate(
            observables,
            vectors,
            KUNTZ_SEED_EPSILONS,
        ).value
        relative_error = abs(value - direct) / max(abs(direct), 1e-30)
        entries.append(
            MixedSpectralConvergenceEntry(
                observables=observables,
                lmax=int(lmax),
                value=value,
                direct=direct,
                relative_error=float(relative_error),
            )
        )
    return MixedSpectralConvergenceResult(observables=observables, entries=tuple(entries))
