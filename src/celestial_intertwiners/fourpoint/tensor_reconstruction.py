"""Helicity-basis reconstruction of real vector and bitensor kernels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..astrometry_response import astrometric_response_vector
from ..finite_kernel_check import canonical_helicity_dyad
from ..geometry import spherical_angles_from_direction, tangent_basis
from .pointwise_descendant import explicit_pta_antenna, helicity_polarization_from_axis


HELICITY_SIGNS = (1, -1)


@dataclass(frozen=True)
class TensorReconstructionAuditResult:
    """Summary of helicity-to-real reconstruction checks."""

    passed: bool
    trials: int
    checked: int
    max_polarization_error: float
    max_pta_error: float
    max_vector_error: float
    max_bitensor_error: float


def helicity_to_real_vector(a_plus, a_minus):
    """Return ``(theta, phi)`` components from m_+/m_- coefficients.

    The package convention uses
    ``m_+ = (e_theta + i e_phi)/sqrt(2)`` and
    ``m_- = (e_theta - i e_phi)/sqrt(2)``. Thus
    ``v = a_plus m_+ + a_minus m_-`` gives the components below. For a real
    vector, ``a_minus`` is the complex conjugate of ``a_plus``.
    """

    return np.array(
        [
            (a_plus + a_minus) / np.sqrt(2.0),
            (a_minus - a_plus) / (1j * np.sqrt(2.0)),
        ],
        dtype=complex,
    )


def helicity_components_from_real_vector(theta_component, phi_component) -> tuple[complex, complex]:
    """Return m_+/m_- coefficients from orthonormal tangent components."""

    a_plus = (theta_component - 1j * phi_component) / np.sqrt(2.0)
    a_minus = (theta_component + 1j * phi_component) / np.sqrt(2.0)
    return complex(a_plus), complex(a_minus)


def tangent_vector_from_components(direction, theta_component, phi_component) -> np.ndarray:
    """Build a Cartesian tangent vector from orthonormal tangent components."""

    theta, phi = spherical_angles_from_direction(direction)
    e_theta, e_phi = tangent_basis(theta, phi)
    return theta_component * e_theta + phi_component * e_phi


def tangent_components_from_vector(direction, vector) -> tuple[complex, complex]:
    """Project a Cartesian tangent vector onto the local orthonormal basis."""

    theta, phi = spherical_angles_from_direction(direction)
    e_theta, e_phi = tangent_basis(theta, phi)
    vector = np.asarray(vector, dtype=complex)
    return complex(np.dot(vector, e_theta)), complex(np.dot(vector, e_phi))


def tangent_vector_from_helicity_components(direction, a_plus, a_minus) -> np.ndarray:
    """Reconstruct a Cartesian tangent vector from m_+/m_- coefficients."""

    return (
        a_plus * canonical_helicity_dyad(direction, sign=1)
        + a_minus * canonical_helicity_dyad(direction, sign=-1)
    )


def helicity_components_from_tangent_vector(direction, vector) -> tuple[complex, complex]:
    """Return m_+/m_- coefficients of a Cartesian tangent vector.

    Because ``m_+ . m_- = 1`` and ``m_+ . m_+ = m_- . m_- = 0`` under the
    bilinear dot product used by the response formulas, the dual of ``m_+`` is
    ``m_-`` and vice versa.
    """

    vector = np.asarray(vector, dtype=complex)
    a_plus = np.dot(vector, canonical_helicity_dyad(direction, sign=-1))
    a_minus = np.dot(vector, canonical_helicity_dyad(direction, sign=1))
    return complex(a_plus), complex(a_minus)


def helicity_components_from_bitensor(direction_a, direction_b, bitensor) -> dict[tuple[int, int], complex]:
    """Return helicity coefficients of a rank-2 tangent bitensor."""

    bitensor = np.asarray(bitensor, dtype=complex)
    components: dict[tuple[int, int], complex] = {}
    for sign_a in HELICITY_SIGNS:
        dual_a = canonical_helicity_dyad(direction_a, sign=-sign_a)
        for sign_b in HELICITY_SIGNS:
            dual_b = canonical_helicity_dyad(direction_b, sign=-sign_b)
            components[(sign_a, sign_b)] = complex(
                np.einsum("i,j,ij->", dual_a, dual_b, bitensor)
            )
    return components


def bitensor_from_helicity_components(
    direction_a,
    direction_b,
    components: dict[tuple[int, int], complex],
) -> np.ndarray:
    """Reconstruct a Cartesian bitensor from helicity coefficients."""

    total = np.zeros((3, 3), dtype=complex)
    for sign_a in HELICITY_SIGNS:
        basis_a = canonical_helicity_dyad(direction_a, sign=sign_a)
        for sign_b in HELICITY_SIGNS:
            basis_b = canonical_helicity_dyad(direction_b, sign=sign_b)
            total += components.get((sign_a, sign_b), 0.0j) * np.outer(basis_a, basis_b)
    return total


def linear_polarizations_from_helicity(axis_direction) -> tuple[np.ndarray, np.ndarray]:
    """Return real plus/cross polarization tensors from circular helicities."""

    h_plus = helicity_polarization_from_axis(axis_direction, 1)
    h_minus = helicity_polarization_from_axis(axis_direction, -1)
    e_plus = (h_plus + h_minus) / 2.0
    e_cross = (h_plus - h_minus) / (2.0j)
    return np.real_if_close(e_plus), np.real_if_close(e_cross)


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("direction must be nonzero")
    return vector / norm


def _random_direction(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    return _unit(vector)


def _random_tangent_vector(rng: np.random.Generator, direction) -> np.ndarray:
    theta, phi = spherical_angles_from_direction(direction)
    e_theta, e_phi = tangent_basis(theta, phi)
    components = rng.normal(size=2)
    return components[0] * e_theta + components[1] * e_phi


def _random_tangent_bitensor(rng: np.random.Generator, direction_a, direction_b) -> np.ndarray:
    theta_a, phi_a = spherical_angles_from_direction(direction_a)
    theta_b, phi_b = spherical_angles_from_direction(direction_b)
    basis_a = tangent_basis(theta_a, phi_a)
    basis_b = tangent_basis(theta_b, phi_b)
    components = rng.normal(size=(2, 2))
    total = np.zeros((3, 3), dtype=float)
    for i, vector_a in enumerate(basis_a):
        for j, vector_b in enumerate(basis_b):
            total += components[i, j] * np.outer(vector_a, vector_b)
    return total


def _relative_norm(left, right) -> float:
    left = np.asarray(left, dtype=complex)
    right = np.asarray(right, dtype=complex)
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-14))


def _pta_linear_antenna(p, omega, polarization) -> complex:
    denominator = 1.0 - float(np.dot(omega, p))
    if abs(denominator) < 1e-12:
        raise ValueError("singular PTA antenna configuration")
    return 0.5 * np.einsum("i,j,ij->", p, p, polarization) / denominator


def _roundtrip_errors(rng: np.random.Generator) -> tuple[float, float]:
    direction_a = _random_direction(rng)
    direction_b = _random_direction(rng)

    vector = _random_tangent_vector(rng, direction_a)
    a_plus, a_minus = helicity_components_from_tangent_vector(direction_a, vector)
    vector_reconstructed = tangent_vector_from_helicity_components(direction_a, a_plus, a_minus)
    component_reconstructed = tangent_vector_from_components(
        direction_a,
        *helicity_to_real_vector(a_plus, a_minus),
    )
    vector_error = max(
        _relative_norm(vector_reconstructed, vector),
        _relative_norm(component_reconstructed, vector),
    )

    bitensor = _random_tangent_bitensor(rng, direction_a, direction_b)
    bitensor_reconstructed = bitensor_from_helicity_components(
        direction_a,
        direction_b,
        helicity_components_from_bitensor(direction_a, direction_b, bitensor),
    )
    return vector_error, _relative_norm(bitensor_reconstructed, bitensor)


def _physical_response_errors(rng: np.random.Generator) -> tuple[float, float, float, float]:
    omega = _random_direction(rng)
    p_a = _random_direction(rng)
    p_b = _random_direction(rng)
    if abs(float(np.dot(omega, p_a))) > 0.86 or abs(float(np.dot(omega, p_b))) > 0.86:
        raise ValueError("near-singular response sample")

    axis = -omega
    h_plus = helicity_polarization_from_axis(axis, 1)
    h_minus = helicity_polarization_from_axis(axis, -1)
    e_plus, e_cross = linear_polarizations_from_helicity(axis)
    polarization_error = max(
        _relative_norm(e_plus + 1j * e_cross, h_plus),
        _relative_norm(e_plus - 1j * e_cross, h_minus),
    )

    theta_a, phi_a = spherical_angles_from_direction(p_a)
    pta_helicity_plus = explicit_pta_antenna(theta_a, phi_a, omega, 1)
    pta_helicity_minus = explicit_pta_antenna(theta_a, phi_a, omega, -1)
    pta_plus = (pta_helicity_plus + pta_helicity_minus) / 2.0
    pta_cross = (pta_helicity_plus - pta_helicity_minus) / (2.0j)
    pta_error = max(
        abs(pta_plus - _pta_linear_antenna(p_a, omega, e_plus))
        / max(abs(pta_plus), 1e-14),
        abs(pta_cross - _pta_linear_antenna(p_a, omega, e_cross))
        / max(abs(pta_cross), 1e-14),
    )

    response_h_plus_a = astrometric_response_vector(p_a, omega, h_plus)
    response_h_minus_a = astrometric_response_vector(p_a, omega, h_minus)
    response_plus_a = (response_h_plus_a + response_h_minus_a) / 2.0
    response_cross_a = (response_h_plus_a - response_h_minus_a) / (2.0j)
    direct_plus_a = astrometric_response_vector(p_a, omega, e_plus)
    direct_cross_a = astrometric_response_vector(p_a, omega, e_cross)
    vector_error = max(
        _relative_norm(response_plus_a, direct_plus_a),
        _relative_norm(response_cross_a, direct_cross_a),
    )
    for response in (direct_plus_a, direct_cross_a, response_h_plus_a, response_h_minus_a):
        a_plus, a_minus = helicity_components_from_tangent_vector(p_a, response)
        reconstructed = tangent_vector_from_helicity_components(p_a, a_plus, a_minus)
        vector_error = max(vector_error, _relative_norm(reconstructed, response))

    response_plus_b = astrometric_response_vector(p_b, omega, e_plus)
    response_cross_b = astrometric_response_vector(p_b, omega, e_cross)
    bitensor = np.outer(direct_plus_a, response_cross_b) + 0.37 * np.outer(direct_cross_a, response_plus_b)
    bitensor_reconstructed = bitensor_from_helicity_components(
        p_a,
        p_b,
        helicity_components_from_bitensor(p_a, p_b, bitensor),
    )
    bitensor_error = _relative_norm(bitensor_reconstructed, bitensor)
    return polarization_error, float(pta_error), vector_error, bitensor_error


@lru_cache(maxsize=8)
def audit_helicity_reconstruction(
    *,
    seed: int = 20260625,
    trials: int = 64,
    tolerance: float = 1e-11,
) -> TensorReconstructionAuditResult:
    """Audit helicity reconstruction for algebraic and physical response samples."""

    rng = np.random.default_rng(seed)
    checked = 0
    attempts = 0
    max_polarization_error = 0.0
    max_pta_error = 0.0
    max_vector_error = 0.0
    max_bitensor_error = 0.0

    while checked < trials and attempts < 12 * trials:
        attempts += 1
        roundtrip_vector_error, roundtrip_bitensor_error = _roundtrip_errors(rng)
        try:
            polarization_error, pta_error, response_vector_error, response_bitensor_error = (
                _physical_response_errors(rng)
            )
        except ValueError:
            continue
        max_polarization_error = max(max_polarization_error, polarization_error)
        max_pta_error = max(max_pta_error, pta_error)
        max_vector_error = max(max_vector_error, roundtrip_vector_error, response_vector_error)
        max_bitensor_error = max(max_bitensor_error, roundtrip_bitensor_error, response_bitensor_error)
        checked += 1

    passed = (
        checked == trials
        and max_polarization_error < tolerance
        and max_pta_error < tolerance
        and max_vector_error < tolerance
        and max_bitensor_error < tolerance
    )
    return TensorReconstructionAuditResult(
        passed=passed,
        trials=trials,
        checked=checked,
        max_polarization_error=max_polarization_error,
        max_pta_error=max_pta_error,
        max_vector_error=max_vector_error,
        max_bitensor_error=max_bitensor_error,
    )


def verify_appp_real_vector_reconstruction(
    *,
    seed: int = 20260625,
    trials: int = 64,
    tolerance: float = 1e-11,
) -> bool:
    """Return True when APPP-level real vector reconstruction passes."""

    result = audit_helicity_reconstruction(seed=seed, trials=trials, tolerance=tolerance)
    return (
        result.checked == trials
        and result.max_polarization_error < tolerance
        and result.max_pta_error < tolerance
        and result.max_vector_error < tolerance
    )


def verify_aapp_real_bitensor_reconstruction(
    *,
    seed: int = 20260625,
    trials: int = 64,
    tolerance: float = 1e-11,
) -> bool:
    """Return True when AAPP-level real bitensor reconstruction passes."""

    result = audit_helicity_reconstruction(seed=seed, trials=trials, tolerance=tolerance)
    return (
        result.checked == trials
        and result.max_polarization_error < tolerance
        and result.max_vector_error < tolerance
        and result.max_bitensor_error < tolerance
    )
