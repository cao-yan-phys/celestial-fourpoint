"""Pointwise explicit antenna-response descendant checks."""

from __future__ import annotations

import math

import numpy as np

from ..astrometry_response import astrometric_response_vector
from ..finite_kernel_check import canonical_helicity_dyad
from ..geometry import sky_direction, spherical_angles_from_direction
from .conventions import validate_epsilon


EXPLICIT_ASTROMETRY_TO_PTA_NORMALIZATION = -2.0 * math.sqrt(2.0)


def _rotation_z_to(direction) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    cos_angle = float(np.dot(z_axis, direction))
    sin_angle = np.linalg.norm(axis)
    if sin_angle < 1e-14:
        return np.eye(3) if cos_angle > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cos_angle) / sin_angle**2)


def helicity_polarization_from_axis(axis_direction, epsilon: int) -> np.ndarray:
    """Return e^+ + i epsilon e^x from a rotation of the z-axis basis."""

    validate_epsilon(epsilon)
    rotation = _rotation_z_to(axis_direction)
    x_axis = rotation @ np.array([1.0, 0.0, 0.0])
    y_axis = rotation @ np.array([0.0, 1.0, 0.0])
    e_plus = np.outer(x_axis, x_axis) - np.outer(y_axis, y_axis)
    e_cross = np.outer(x_axis, y_axis) + np.outer(y_axis, x_axis)
    return e_plus + 1j * epsilon * e_cross


def explicit_pta_antenna(theta: float, phi: float, omega, epsilon: int) -> complex:
    """Return helicity PTA antenna in the calibrated four-point convention.

    ``omega`` is the GW source direction. The GW propagation direction is
    ``-omega``; the circular polarization basis is fixed at ``-omega``.
    """

    validate_epsilon(epsilon)
    p = sky_direction(theta, phi)
    omega = np.asarray(omega, dtype=float)
    polarization = helicity_polarization_from_axis(-omega, epsilon)
    denominator = 1.0 - float(np.dot(omega, p))
    if abs(denominator) < 1e-12:
        raise ValueError("singular PTA antenna configuration")
    return 0.5 * np.einsum("i,j,ij->", p, p, polarization) / denominator


def explicit_astrometry_antenna_raw(theta: float, phi: float, omega, epsilon: int) -> complex:
    """Return raw helicity astrometric antenna in the calibrated convention."""

    validate_epsilon(epsilon)
    p = sky_direction(theta, phi)
    omega = np.asarray(omega, dtype=float)
    polarization = helicity_polarization_from_axis(-omega, epsilon)
    response = astrometric_response_vector(p, omega, polarization)
    # The raw vector is projected onto the spin s=-epsilon component using the
    # matching dyad convention. This is the convention that gives a nontrivial
    # constant descendant factor against the explicit PTA antenna.
    component_dyad = canonical_helicity_dyad(p, sign=-epsilon)
    return np.dot(response, component_dyad)


def explicit_astrometry_antenna(theta: float, phi: float, omega, epsilon: int) -> complex:
    """Return normalized astrometric antenna satisfying D^epsilon A = P."""

    return explicit_astrometry_antenna_raw(theta, phi, omega, epsilon) / (
        EXPLICIT_ASTROMETRY_TO_PTA_NORMALIZATION
    )


def eth_numeric(function, theta: float, phi: float, spin: int, *, step: float = 1e-5) -> complex:
    """Numerically apply Goldberg eth to a spin-weighted function."""

    value = function(theta, phi)
    d_theta = (function(theta + step, phi) - function(theta - step, phi)) / (2 * step)
    d_phi = (function(theta, phi + step) - function(theta, phi - step)) / (2 * step)
    return -(
        d_theta
        + 1j * d_phi / math.sin(theta)
        - spin * value / math.tan(theta)
    )


def bar_eth_numeric(function, theta: float, phi: float, spin: int, *, step: float = 1e-5) -> complex:
    """Numerically apply Goldberg bar-eth to a spin-weighted function."""

    value = function(theta, phi)
    d_theta = (function(theta + step, phi) - function(theta - step, phi)) / (2 * step)
    d_phi = (function(theta, phi + step) - function(theta, phi - step)) / (2 * step)
    return -(
        d_theta
        - 1j * d_phi / math.sin(theta)
        + spin * value / math.tan(theta)
    )


def descendant_operator_numeric(
    function,
    theta: float,
    phi: float,
    epsilon: int,
    *,
    step: float = 1e-5,
) -> complex:
    """Apply the explicit-response descendant operator.

    In this response convention, ``D^+ = eth`` and ``D^- = bar_eth``.
    """

    validate_epsilon(epsilon)
    spin = -epsilon
    if epsilon == 1:
        return eth_numeric(function, theta, phi, spin, step=step)
    return bar_eth_numeric(function, theta, phi, spin, step=step)


def pointwise_descendant_residual(theta: float, phi: float, omega, epsilon: int) -> tuple[complex, complex, float]:
    """Return descendant value, PTA value, and relative error."""

    antenna = lambda th, ph: explicit_astrometry_antenna(th, ph, omega, epsilon)
    descendant = descendant_operator_numeric(antenna, theta, phi, epsilon)
    pta = explicit_pta_antenna(theta, phi, omega, epsilon)
    relative_error = abs(descendant - pta) / max(abs(pta), abs(descendant), 1e-14)
    return descendant, pta, float(relative_error)


def random_pointwise_descendant_check(
    *,
    seed: int = 20260625,
    trials: int = 64,
    tolerance: float = 1e-8,
) -> tuple[bool, float]:
    """Run random nonsingular pointwise descendant checks."""

    rng = np.random.default_rng(seed)
    max_error = 0.0
    checked = 0
    attempts = 0
    while checked < trials and attempts < 10 * trials:
        attempts += 1
        omega = rng.normal(size=3)
        omega /= np.linalg.norm(omega)
        p = rng.normal(size=3)
        p /= np.linalg.norm(p)
        if abs(float(np.dot(p, omega))) > 0.82 or abs(p[2]) > 0.92:
            continue
        theta, phi = spherical_angles_from_direction(p)
        for epsilon in (-1, 1):
            _descendant, _pta, error = pointwise_descendant_residual(theta, phi, omega, epsilon)
            max_error = max(max_error, error)
        checked += 1
    return checked == trials and max_error < tolerance, max_error
