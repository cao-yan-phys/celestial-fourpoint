"""Standard Earth-term astrometric response checks."""

from __future__ import annotations

import numpy as np

from .geometry import orthonormal_components, sky_direction


X_HAT = np.array([1.0, 0.0, 0.0])
Y_HAT = np.array([0.0, 1.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])


def plus_polarization_z() -> np.ndarray:
    """Return e^+ for a wave direction q = z."""

    return np.outer(X_HAT, X_HAT) - np.outer(Y_HAT, Y_HAT)


def cross_polarization_z() -> np.ndarray:
    """Return e^x for a wave direction q = z."""

    return np.outer(X_HAT, Y_HAT) + np.outer(Y_HAT, X_HAT)


def astrometric_response_vector(n, q, polarization) -> np.ndarray:
    """Return the Earth-term astrometric response vector.

    Parameters use Cartesian components. The result is tangent to ``n`` away
    from the singular line ``q . n = 1``.
    """

    n = np.asarray(n, dtype=complex)
    q = np.asarray(q, dtype=complex)
    e = np.asarray(polarization, dtype=complex)
    denom = 1 - np.dot(q, n)
    if abs(denom) < 1e-14:
        raise ValueError("singular configuration: q . n is too close to 1")
    nn_e = np.einsum("j,k,jk->", n, n, e)
    e_n = e @ n
    return 0.5 * (((n - q) / denom) * nn_e - e_n)


def q_z_response_components(theta, phi, polarization: str):
    """Return orthonormal (theta, phi) response components for q = z."""

    pol = polarization.upper()
    if pol in {"+", "PLUS"}:
        e = plus_polarization_z()
    elif pol in {"X", "CROSS", "TIMES"}:
        e = cross_polarization_z()
    else:
        raise ValueError("polarization must be plus or cross")
    n = sky_direction(theta, phi)
    response = astrometric_response_vector(n, Z_HAT, e)
    return orthonormal_components(response.real, theta, phi)


def q_z_expected_components(theta, phi, polarization: str):
    """Analytic q = z components from the audit conventions."""

    pol = polarization.upper()
    if pol in {"+", "PLUS"}:
        return (
            0.5 * np.sin(theta) * np.cos(2 * phi),
            0.5 * np.sin(theta) * np.sin(2 * phi),
        )
    if pol in {"X", "CROSS", "TIMES"}:
        return (
            0.5 * np.sin(theta) * np.sin(2 * phi),
            -0.5 * np.sin(theta) * np.cos(2 * phi),
        )
    raise ValueError("polarization must be plus or cross")


def circular_q_z_components(theta, phi):
    """Return R^+ + i R^x in the orthonormal tangent basis."""

    plus_theta, plus_phi = q_z_response_components(theta, phi, "plus")
    cross_theta, cross_phi = q_z_response_components(theta, phi, "cross")
    return plus_theta + 1j * cross_theta, plus_phi + 1j * cross_phi


def circular_q_z_expected_components(theta, phi):
    """Expected circular response in the q = z convention."""

    phase = np.exp(2j * phi)
    amplitude = 0.5 * np.sin(theta) * phase
    return amplitude, -1j * amplitude


def tangency_residual(n, q, polarization) -> complex:
    """Return n_i R^i for a response vector."""

    response = astrometric_response_vector(n, q, polarization)
    return np.dot(np.asarray(n, dtype=complex), response)
