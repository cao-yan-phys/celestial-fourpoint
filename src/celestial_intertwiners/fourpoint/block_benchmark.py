"""Numerical finite-block benchmarks for four-point kernels."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..geometry import spherical_angles_from_direction, sky_direction
from ..invariant_tensors import wigner_3j
from .conventions import angular_output_spin, helicity_selection_rule, normalize_observable, validate_epsilon
from .descendants import pppp_to_mixed_inverse_factor
from .direct_quadrature import gauss_legendre_sphere_integral, spin_weighted_spherical_harmonic
from .helicity_responses import response_transfer
from .kuntz_benchmark import KUNTZ_SEED_EPSILONS, load_kuntz_fixtures
from .coupled_blocks import intermediate_L_values
from .spin_gaunt import spin_gaunt_prefactor


@dataclass(frozen=True)
class BlockDirectBenchmarkResult:
    """Finite spectral PPPP block-vs-direct comparison."""

    lmax: int
    block: complex
    direct: complex
    relative_error: float
    passed: bool


@dataclass(frozen=True)
class MixedDescendantBenchmarkResult:
    """Finite mixed-kernel descendant-vs-direct comparison."""

    observables: tuple[str, str, str, str]
    lmax: int
    descendant: complex
    direct: complex
    relative_error: float
    passed: bool


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if norm == 0:
        raise ValueError("direction vector must be nonzero")
    return array / norm


def _unit_vectors(vectors) -> np.ndarray:
    if len(vectors) != 4:
        raise ValueError("four-point benchmark requires four directions")
    return np.asarray([_unit(vector) for vector in vectors], dtype=float)


def _validate_epsilons(epsilons: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if len(epsilons) != 4:
        raise ValueError("PPPP benchmark requires four helicities")
    epsilons = tuple(validate_epsilon(epsilon) for epsilon in epsilons)
    if not helicity_selection_rule(epsilons):
        raise ValueError("finite PPPP benchmark requires an allowed two-plus/two-minus assignment")
    return epsilons


def _validate_observables(observables) -> tuple[str, str, str, str]:
    if len(observables) != 4:
        raise ValueError("four-point benchmark requires four observables")
    normalized = tuple(normalize_observable(observable) for observable in observables)
    if any(observable not in {"P", "A"} for observable in normalized):
        raise ValueError("finite angular benchmark supports only P and A observables")
    return normalized


@lru_cache(maxsize=None)
def _wigner_3j_float(l1: int, l2: int, l3: int, m1: int, m2: int, m3: int) -> float:
    return float(wigner_3j(l1, l2, l3, m1, m2, m3).evalf())


@lru_cache(maxsize=None)
def _spin_gaunt_float(l1: int, l2: int, L: int, s1: int, s2: int, s3: int) -> float:
    return float(spin_gaunt_prefactor(l1, l2, L, s1, s2, s3).evalf())


@lru_cache(maxsize=None)
def _source_integral_block_float(
    ells: tuple[int, int, int, int],
    ms: tuple[int, int, int, int],
    epsilons: tuple[int, int, int, int],
    pairing: tuple[int, int, int, int],
) -> float:
    """Return the source-side block coefficient as a float."""

    if sum(ms) != 0:
        return 0.0
    i1, i2, i3, i4 = pairing
    l1, l2, l3, l4 = (ells[i1], ells[i2], ells[i3], ells[i4])
    m1, m2, m3, m4 = (ms[i1], ms[i2], ms[i3], ms[i4])
    spins = tuple(2 * epsilon for epsilon in epsilons)
    s1, s2, s3, s4 = (spins[i1], spins[i2], spins[i3], spins[i4])
    source_spin_sum = s1 + s2
    if source_spin_sum + s3 + s4 != 0:
        return 0.0

    total = 0.0
    for L in intermediate_L_values(ells, pairing):
        g12 = _spin_gaunt_float(l1, l2, L, s1, s2, -source_spin_sum)
        g34 = _spin_gaunt_float(l3, l4, L, s3, s4, source_spin_sum)
        if g12 == 0.0 or g34 == 0.0:
            continue
        for M in range(-L, L + 1):
            total += (
                (-1) ** (M + source_spin_sum)
                * g12
                * g34
                * _wigner_3j_float(l1, l2, L, m1, m2, M)
                * _wigner_3j_float(l3, l4, L, m3, m4, -M)
            )
    return total


def truncated_pta_response(
    p_vector,
    omega,
    epsilon: int,
    *,
    lmax: int,
    transfer_scale: float = 1.0,
) -> complex:
    """Evaluate the finite bi-spin PTA response used by the block benchmark."""

    return truncated_bispin_response(
        "P",
        p_vector,
        omega,
        epsilon,
        lmax=lmax,
        transfer_scale=transfer_scale,
    )


def truncated_bispin_response(
    observable: str,
    p_vector,
    omega,
    epsilon: int,
    *,
    lmax: int,
    transfer_scale: float = 1.0,
) -> complex:
    """Evaluate a finite bi-spin P/A response in the benchmark convention."""

    if lmax < 2:
        raise ValueError("lmax must be at least 2")
    observable = normalize_observable(observable)
    if observable not in {"P", "A"}:
        raise ValueError("finite angular benchmark supports only P and A observables")
    epsilon = validate_epsilon(epsilon)
    theta_p, phi_p = spherical_angles_from_direction(_unit(p_vector))
    theta_omega, phi_omega = spherical_angles_from_direction(_unit(omega))
    external_spin = angular_output_spin(observable, epsilon)
    total = 0.0j
    for ell in range(2, lmax + 1):
        transfer = transfer_scale * response_transfer(observable, ell, exact=False)
        for m in range(-ell, ell + 1):
            total += (
                transfer
                * spin_weighted_spherical_harmonic(ell, m, external_spin, theta_p, phi_p)
                * spin_weighted_spherical_harmonic(ell, m, 2 * epsilon, theta_omega, phi_omega)
            )
    return total


def pppp_truncated_direct_average(
    vectors,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    lmax: int = 3,
    n_theta: int = 16,
    n_phi: int = 32,
    transfer_scale: float = 1.0,
) -> complex:
    """Directly integrate the finite spectral PPPP kernel over source direction."""

    unit_vectors = _unit_vectors(vectors)
    epsilons = _validate_epsilons(epsilons)

    def integrand(theta: float, phi: float) -> complex:
        omega = sky_direction(theta, phi)
        value = 1.0 + 0.0j
        for vector, epsilon in zip(unit_vectors, epsilons):
            value *= truncated_pta_response(
                vector,
                omega,
                epsilon,
                lmax=lmax,
                transfer_scale=transfer_scale,
            )
        return value

    return gauss_legendre_sphere_integral(
        integrand, n_theta=n_theta, n_phi=n_phi
    ) / (4.0 * math.pi)


def mixed_truncated_direct_average(
    observables,
    vectors,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    lmax: int = 3,
    n_theta: int = 16,
    n_phi: int = 32,
    transfer_scale: float = 1.0,
) -> complex:
    """Directly integrate a finite spectral mixed P/A kernel."""

    observables = _validate_observables(observables)
    unit_vectors = _unit_vectors(vectors)
    epsilons = _validate_epsilons(epsilons)

    def integrand(theta: float, phi: float) -> complex:
        omega = sky_direction(theta, phi)
        value = 1.0 + 0.0j
        for observable, vector, epsilon in zip(observables, unit_vectors, epsilons):
            value *= truncated_bispin_response(
                observable,
                vector,
                omega,
                epsilon,
                lmax=lmax,
                transfer_scale=transfer_scale,
            )
        return value

    return gauss_legendre_sphere_integral(
        integrand, n_theta=n_theta, n_phi=n_phi
    ) / (4.0 * math.pi)


def pppp_coupled_block_average(
    vectors,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    lmax: int = 3,
    transfer_scale: float = 1.0,
    pairing: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> complex:
    """Evaluate the finite spectral PPPP kernel with coupled source blocks."""

    if lmax < 2:
        raise ValueError("lmax must be at least 2")
    unit_vectors = _unit_vectors(vectors)
    epsilons = _validate_epsilons(epsilons)
    angles = [spherical_angles_from_direction(vector) for vector in unit_vectors]
    external_y: dict[tuple[int, int, int], complex] = {}
    for leg_index, (theta, phi) in enumerate(angles):
        for ell in range(2, lmax + 1):
            for m in range(-ell, ell + 1):
                external_y[(leg_index, ell, m)] = spin_weighted_spherical_harmonic(
                    ell, m, 0, theta, phi
                )

    total = 0.0j
    ell_values = range(2, lmax + 1)
    for ells in itertools.product(ell_values, repeat=4):
        transfer = 1.0
        for ell in ells:
            transfer *= transfer_scale * response_transfer("P", ell, exact=False)
        m_ranges = [range(-ell, ell + 1) for ell in ells]
        for ms in itertools.product(*m_ranges):
            source = _source_integral_block_float(ells, ms, epsilons, pairing)
            if source == 0.0:
                continue
            external = 1.0 + 0.0j
            for leg_index, (ell, m) in enumerate(zip(ells, ms)):
                external *= external_y[(leg_index, ell, m)]
            total += transfer * source * external
    return total / (4.0 * math.pi)


def mixed_descendant_block_average(
    observables,
    vectors,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    lmax: int = 3,
    transfer_scale: float = 1.0,
    pairing: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> complex:
    """Generate a finite mixed kernel from the PPPP block by inverse descendants."""

    if lmax < 2:
        raise ValueError("lmax must be at least 2")
    observables = _validate_observables(observables)
    unit_vectors = _unit_vectors(vectors)
    epsilons = _validate_epsilons(epsilons)
    angles = [spherical_angles_from_direction(vector) for vector in unit_vectors]
    external_y: dict[tuple[int, int, int], complex] = {}
    for leg_index, (observable, epsilon, (theta, phi)) in enumerate(
        zip(observables, epsilons, angles)
    ):
        spin = angular_output_spin(observable, epsilon)
        for ell in range(2, lmax + 1):
            for m in range(-ell, ell + 1):
                external_y[(leg_index, ell, m)] = spin_weighted_spherical_harmonic(
                    ell, m, spin, theta, phi
                )

    total = 0.0j
    ell_values = range(2, lmax + 1)
    for ells in itertools.product(ell_values, repeat=4):
        transfer = 1.0
        for ell in ells:
            transfer *= transfer_scale * response_transfer("P", ell, exact=False)
        transfer *= pppp_to_mixed_inverse_factor(observables, epsilons, ells, exact=False)
        m_ranges = [range(-ell, ell + 1) for ell in ells]
        for ms in itertools.product(*m_ranges):
            source = _source_integral_block_float(ells, ms, epsilons, pairing)
            if source == 0.0:
                continue
            external = 1.0 + 0.0j
            for leg_index, (ell, m) in enumerate(zip(ells, ms)):
                external *= external_y[(leg_index, ell, m)]
            total += transfer * source * external
    return total / (4.0 * math.pi)


def benchmark_pppp_block_vs_direct(
    vectors=None,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    lmax: int = 3,
    tolerance: float = 1e-11,
    n_theta: int = 16,
    n_phi: int = 32,
    transfer_scale: float = 1.0,
) -> BlockDirectBenchmarkResult:
    """Compare finite PPPP coupled-block evaluation with direct quadrature."""

    if vectors is None:
        vectors = load_kuntz_fixtures()[0].vectors
    block = pppp_coupled_block_average(
        vectors,
        epsilons,
        lmax=lmax,
        transfer_scale=transfer_scale,
    )
    direct = pppp_truncated_direct_average(
        vectors,
        epsilons,
        lmax=lmax,
        n_theta=n_theta,
        n_phi=n_phi,
        transfer_scale=transfer_scale,
    )
    relative_error = abs(block - direct) / max(abs(direct), 1e-30)
    return BlockDirectBenchmarkResult(
        lmax=lmax,
        block=block,
        direct=direct,
        relative_error=float(relative_error),
        passed=bool(relative_error < tolerance),
    )


def benchmark_mixed_descendant_vs_direct(
    observables,
    vectors=None,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    lmax: int = 3,
    tolerance: float = 1e-11,
    n_theta: int = 16,
    n_phi: int = 32,
    transfer_scale: float = 1.0,
) -> MixedDescendantBenchmarkResult:
    """Compare inverse-descendant mixed block evaluation with direct quadrature."""

    observables = _validate_observables(observables)
    if vectors is None:
        vectors = load_kuntz_fixtures()[0].vectors
    descendant = mixed_descendant_block_average(
        observables,
        vectors,
        epsilons,
        lmax=lmax,
        transfer_scale=transfer_scale,
    )
    direct = mixed_truncated_direct_average(
        observables,
        vectors,
        epsilons,
        lmax=lmax,
        n_theta=n_theta,
        n_phi=n_phi,
        transfer_scale=transfer_scale,
    )
    relative_error = abs(descendant - direct) / max(abs(direct), 1e-30)
    return MixedDescendantBenchmarkResult(
        observables=observables,
        lmax=lmax,
        descendant=descendant,
        direct=direct,
        relative_error=float(relative_error),
        passed=bool(relative_error < tolerance),
    )


@lru_cache(maxsize=8)
def verify_pppp_block_vs_direct(
    *,
    lmax: int = 3,
    tolerance: float = 1e-11,
    n_theta: int = 16,
    n_phi: int = 32,
) -> bool:
    """Return True when the finite PPPP block sum matches direct quadrature."""

    return benchmark_pppp_block_vs_direct(
        lmax=lmax,
        tolerance=tolerance,
        n_theta=n_theta,
        n_phi=n_phi,
    ).passed


@lru_cache(maxsize=8)
def verify_appp_descendant_vs_direct(
    *,
    lmax: int = 3,
    tolerance: float = 1e-11,
    n_theta: int = 16,
    n_phi: int = 32,
) -> bool:
    """Return True when finite APPP descendant construction matches direct quadrature."""

    return benchmark_mixed_descendant_vs_direct(
        ("A", "P", "P", "P"),
        lmax=lmax,
        tolerance=tolerance,
        n_theta=n_theta,
        n_phi=n_phi,
    ).passed


@lru_cache(maxsize=8)
def verify_aapp_descendant_vs_direct(
    *,
    lmax: int = 3,
    tolerance: float = 1e-11,
    n_theta: int = 16,
    n_phi: int = 32,
) -> bool:
    """Return True when finite AAPP descendant construction matches direct quadrature."""

    return benchmark_mixed_descendant_vs_direct(
        ("A", "A", "P", "P"),
        lmax=lmax,
        tolerance=tolerance,
        n_theta=n_theta,
        n_phi=n_phi,
    ).passed


@lru_cache(maxsize=8)
def verify_aaap_descendant_vs_direct(
    *,
    lmax: int = 3,
    tolerance: float = 1e-11,
    n_theta: int = 16,
    n_phi: int = 32,
) -> bool:
    """Return True when finite AAAP descendant construction matches direct quadrature."""

    return benchmark_mixed_descendant_vs_direct(
        ("A", "A", "A", "P"),
        lmax=lmax,
        tolerance=tolerance,
        n_theta=n_theta,
        n_phi=n_phi,
    ).passed


@lru_cache(maxsize=8)
def verify_aaaa_descendant_vs_direct(
    *,
    lmax: int = 3,
    tolerance: float = 1e-11,
    n_theta: int = 16,
    n_phi: int = 32,
) -> bool:
    """Return True when finite AAAA descendant construction matches direct quadrature."""

    return benchmark_mixed_descendant_vs_direct(
        ("A", "A", "A", "A"),
        lmax=lmax,
        tolerance=tolerance,
        n_theta=n_theta,
        n_phi=n_phi,
    ).passed


@lru_cache(maxsize=8)
def verify_finite_block_higher_cutoff(
    *,
    lmax: int = 4,
    tolerance: float = 1e-10,
    n_theta: int = 10,
    n_phi: int = 20,
) -> bool:
    """Check PPPP/APPP/AAPP finite block-vs-direct closure beyond the minimal cutoff."""

    return (
        benchmark_pppp_block_vs_direct(
            lmax=lmax,
            tolerance=tolerance,
            n_theta=n_theta,
            n_phi=n_phi,
        ).passed
        and benchmark_mixed_descendant_vs_direct(
            ("A", "P", "P", "P"),
            lmax=lmax,
            tolerance=tolerance,
            n_theta=n_theta,
            n_phi=n_phi,
        ).passed
        and benchmark_mixed_descendant_vs_direct(
            ("A", "A", "P", "P"),
            lmax=lmax,
            tolerance=tolerance,
            n_theta=n_theta,
            n_phi=n_phi,
        ).passed
    )


@lru_cache(maxsize=8)
def verify_pppp_permutation_symmetry(
    *,
    lmax: int = 3,
    tolerance: float = 1e-12,
) -> bool:
    """Check the Kuntz seed permutation pattern with finite PPPP blocks."""

    fixtures = load_kuntz_fixtures()
    if len(fixtures) < 4:
        return False
    values = tuple(
        pppp_coupled_block_average(fixture.vectors, KUNTZ_SEED_EPSILONS, lmax=lmax)
        for fixture in fixtures[:4]
    )
    scale = max(abs(values[0]), 1e-30)
    return (
        abs(values[1] - values[0]) / scale < tolerance
        and abs(values[2] - values[0]) / scale < tolerance
        and abs(values[3] - values[0].conjugate()) / scale < tolerance
    )


@lru_cache(maxsize=8)
def verify_mixed_kernel_permutation_symmetry(
    *,
    lmax: int = 3,
    tolerance: float = 1e-12,
) -> bool:
    """Check that mixed finite block kernels are invariant under relabeling."""

    vectors = load_kuntz_fixtures()[0].vectors
    observables = ("A", "P", "P", "P")
    epsilons = KUNTZ_SEED_EPSILONS
    base = mixed_descendant_block_average(observables, vectors, epsilons, lmax=lmax)
    scale = max(abs(base), 1e-30)
    permutations = ((0, 1, 3, 2), (0, 2, 1, 3), (1, 0, 2, 3), (2, 1, 0, 3))
    for permutation in permutations:
        permuted_observables = tuple(observables[index] for index in permutation)
        permuted_epsilons = tuple(epsilons[index] for index in permutation)
        permuted_vectors = [vectors[index] for index in permutation]
        value = mixed_descendant_block_average(
            permuted_observables,
            permuted_vectors,
            permuted_epsilons,
            lmax=lmax,
        )
        if abs(value - base) / scale >= tolerance:
            return False
    return True
