"""Finite Lorentz covariance checks for the astrometric response kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .astrometry_response import astrometric_response_vector
from .geometry import tangent_basis, spherical_angles_from_direction


@dataclass(frozen=True)
class FiniteKernelCheckResult:
    """Summary of a finite-kernel covariance audit."""

    passed: bool
    trials: int
    checked: int
    skipped: int
    max_relative_error: float
    max_absolute_error: float


def canonical_helicity_dyad(direction, sign: int = 1) -> np.ndarray:
    """Return m_+ or m_- at a sky direction.

    ``sign=+1`` gives ``(e_theta + i e_phi)/sqrt(2)`` and ``sign=-1`` gives
    ``(e_theta - i e_phi)/sqrt(2)``. These are spatial tangent vectors in the
    orthonormal sky frame.
    """

    if sign not in {1, -1}:
        raise ValueError("dyad sign must be +1 or -1")
    theta, phi = spherical_angles_from_direction(direction)
    e_theta, e_phi = tangent_basis(theta, phi)
    return (e_theta + sign * 1j * e_phi) / np.sqrt(2.0)


def helicity_polarization(direction, helicity: int = 2) -> np.ndarray:
    """Return the tensor helicity polarization dyad outer product."""

    if helicity == 2:
        dyad = canonical_helicity_dyad(direction, sign=1)
    elif helicity == -2:
        dyad = canonical_helicity_dyad(direction, sign=-1)
    else:
        raise ValueError("tensor helicity must be +2 or -2")
    return np.outer(dyad, dyad)


def astrometric_kernel_helicity_component(
    n,
    q,
    *,
    input_helicity: int = 2,
    output_spin: int = -1,
) -> complex:
    """Return the helicity component of the astrometric response kernel.

    The finite covariance audit uses the ``+2 -> -1`` channel.  With the
    response convention of ``CODEX_ASTROMETRY_LORENTZ_AUDIT.md``, the q=z
    circular response is proportional to ``e_theta - i e_phi``.  Its spin
    component is extracted as ``R_theta + i R_phi``, equivalently the bilinear
    dot product with ``m_+(n)``.  Thus the component extraction dyad has sign
    ``-output_spin``.
    """

    if output_spin not in {1, -1}:
        raise ValueError("output spin must be +1 or -1")
    polarization = helicity_polarization(q, input_helicity)
    response = astrometric_response_vector(n, q, polarization)
    output_dyad = canonical_helicity_dyad(n, sign=-output_spin)
    return np.dot(response, output_dyad)


def boost_matrix(beta) -> np.ndarray:
    """Return a proper orthochronous boost matrix for velocity vector beta."""

    beta = np.asarray(beta, dtype=float)
    beta_norm = np.linalg.norm(beta)
    if beta_norm >= 1:
        raise ValueError("boost velocity must satisfy |beta| < 1")
    matrix = np.eye(4)
    if beta_norm == 0:
        return matrix
    beta_hat = beta / beta_norm
    gamma = 1.0 / np.sqrt(1.0 - beta_norm**2)
    matrix[0, 0] = gamma
    matrix[0, 1:] = gamma * beta
    matrix[1:, 0] = gamma * beta
    matrix[1:, 1:] += (gamma - 1.0) * np.outer(beta_hat, beta_hat)
    return matrix


def rotation_matrix_3(axis, angle: float) -> np.ndarray:
    """Return a 3D right-handed rotation matrix."""

    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("rotation axis must be nonzero")
    x, y, z = axis / axis_norm
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def spatial_rotation_lorentz(rotation_3: np.ndarray) -> np.ndarray:
    """Embed a spatial rotation as a Lorentz matrix."""

    matrix = np.eye(4)
    matrix[1:, 1:] = np.asarray(rotation_3, dtype=float)
    return matrix


def random_lorentz_matrix(rng: np.random.Generator, *, max_beta: float = 0.45) -> np.ndarray:
    """Generate a random proper orthochronous Lorentz matrix."""

    axis_1 = rng.normal(size=3)
    axis_2 = rng.normal(size=3)
    angle_1 = rng.uniform(-np.pi, np.pi)
    angle_2 = rng.uniform(-np.pi, np.pi)
    beta_axis = rng.normal(size=3)
    beta_axis /= np.linalg.norm(beta_axis)
    beta_min = min(0.02, 0.2 * max_beta)
    beta = rng.uniform(beta_min, max_beta) * beta_axis
    return (
        spatial_rotation_lorentz(rotation_matrix_3(axis_2, angle_2))
        @ boost_matrix(beta)
        @ spatial_rotation_lorentz(rotation_matrix_3(axis_1, angle_1))
    )


def transform_null_direction(lorentz: np.ndarray, direction) -> tuple[np.ndarray, float]:
    """Apply a Lorentz matrix to k=(1,n) and return (n', Omega)."""

    null = np.concatenate(([1.0], np.asarray(direction, dtype=float)))
    transformed = np.asarray(lorentz, dtype=float) @ null
    omega = float(transformed[0])
    if omega <= 0:
        raise ValueError("Lorentz transform must be proper orthochronous on this null ray")
    direction_prime = transformed[1:] / omega
    direction_prime = direction_prime / np.linalg.norm(direction_prime)
    return direction_prime, omega


def pushforward_screen_vector(lorentz: np.ndarray, direction, tangent_vector) -> tuple[np.ndarray, np.ndarray, float]:
    """Push a tangent vector through a Lorentz map and return a screen vector.

    The transformed four-vector generally has a time component. Subtracting
    that time component times the transformed null direction fixes the screen
    representative with zero time component.
    """

    direction_prime, omega = transform_null_direction(lorentz, direction)
    vector_4 = np.concatenate(([0.0], np.asarray(tangent_vector, dtype=complex)))
    pushed = np.asarray(lorentz, dtype=float).astype(complex) @ vector_4
    screen = pushed[1:] - pushed[0] * direction_prime
    return screen, direction_prime, omega


def wigner_phase(lorentz: np.ndarray, direction, *, dyad_sign: int) -> complex:
    """Return phase defined by ``g_* m = phase * m'`` in screen gauge."""

    dyad = canonical_helicity_dyad(direction, sign=dyad_sign)
    screen, direction_prime, omega = pushforward_screen_vector(lorentz, direction, dyad)
    canonical_prime = canonical_helicity_dyad(direction_prime, sign=dyad_sign)
    phase = np.vdot(canonical_prime, screen)
    magnitude = abs(phase)
    if magnitude == 0:
        raise ValueError("degenerate Wigner phase extraction")
    return phase / magnitude


def finite_kernel_covariance_residual(n, q, lorentz: np.ndarray) -> tuple[complex, complex, float, float]:
    """Return lhs, rhs, absolute error, and relative error for the finite law.

    The checked law is

    ``R'_-(g n, g q) = Omega_n phase_n^{-1} phase_q^2 R_-(n, q)``.

    There is no ``Omega_q`` factor in this kernel law; the input density weight
    is canceled by the transformed source-direction measure.
    """

    n_prime, omega_n = transform_null_direction(lorentz, n)
    q_prime, _omega_q = transform_null_direction(lorentz, q)
    lhs = astrometric_kernel_helicity_component(n_prime, q_prime)
    old = astrometric_kernel_helicity_component(n, q)
    phase_n = wigner_phase(lorentz, n, dyad_sign=1)
    # The audit's input helicity phase is the conjugate of the m_+ dyad phase
    # under the canonical spherical-basis convention used here.
    phase_q = np.conj(wigner_phase(lorentz, q, dyad_sign=1))
    rhs = omega_n * phase_n ** (-1) * phase_q**2 * old
    absolute = float(abs(lhs - rhs))
    relative = absolute / max(float(abs(lhs)), float(abs(rhs)), 1e-14)
    return lhs, rhs, absolute, relative


def _random_direction(rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=3)
    return direction / np.linalg.norm(direction)


def _configuration_is_safe(n, q, lorentz: np.ndarray, *, min_separation: float = 5e-3) -> bool:
    if 1.0 - abs(float(np.dot(n, q))) < min_separation:
        return False
    n_prime, _ = transform_null_direction(lorentz, n)
    q_prime, _ = transform_null_direction(lorentz, q)
    if 1.0 - abs(float(np.dot(n_prime, q_prime))) < min_separation:
        return False
    return True


def finite_kernel_covariance_check(
    *,
    seed: int = 20260623,
    trials: int = 96,
    tolerance: float = 1e-9,
    max_beta: float = 0.45,
) -> FiniteKernelCheckResult:
    """Run a deterministic random finite-kernel covariance check."""

    rng = np.random.default_rng(seed)
    checked = 0
    skipped = 0
    max_relative = 0.0
    max_absolute = 0.0

    attempts = 0
    max_attempts = trials * 8
    while checked < trials and attempts < max_attempts:
        attempts += 1
        lorentz = random_lorentz_matrix(rng, max_beta=max_beta)
        n = _random_direction(rng)
        q = _random_direction(rng)
        if not _configuration_is_safe(n, q, lorentz):
            skipped += 1
            continue
        _lhs, _rhs, absolute, relative = finite_kernel_covariance_residual(n, q, lorentz)
        max_absolute = max(max_absolute, absolute)
        max_relative = max(max_relative, relative)
        checked += 1

    passed = checked == trials and max_relative < tolerance
    return FiniteKernelCheckResult(
        passed=passed,
        trials=trials,
        checked=checked,
        skipped=skipped,
        max_relative_error=max_relative,
        max_absolute_error=max_absolute,
    )
