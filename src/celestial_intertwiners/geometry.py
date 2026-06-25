"""Elementary sky geometry."""

from __future__ import annotations

import numpy as np


def sky_direction(theta, phi):
    """Return n_hat(theta, phi) in Cartesian coordinates."""

    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def tangent_basis(theta, phi):
    """Return the orthonormal (e_theta, e_phi) basis at a sky direction."""

    e_theta = np.array(
        [
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ]
    )
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0])
    return e_theta, e_phi


def spherical_angles_from_direction(direction):
    """Return (theta, phi) for a unit 3-vector."""

    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("direction must be nonzero")
    unit = direction / norm
    theta = np.arccos(np.clip(unit[2], -1.0, 1.0))
    phi = np.arctan2(unit[1], unit[0])
    return theta, phi


def orthonormal_components(vector, theta, phi):
    """Project a 3-vector onto the orthonormal tangent basis."""

    e_theta, e_phi = tangent_basis(theta, phi)
    return np.dot(vector, e_theta), np.dot(vector, e_phi)


def sigma_from_cosine(cos_theta):
    """Return sigma = (1 - cos(theta)) / 2."""

    return (1 - cos_theta) / 2


def cosine_from_sigma(sigma):
    """Return cos(theta) = 1 - 2 sigma."""

    return 1 - 2 * sigma


def sigma_between(theta1, phi1, theta2, phi2):
    """Return sigma for two sky directions."""

    n1 = sky_direction(theta1, phi1)
    n2 = sky_direction(theta2, phi2)
    return sigma_from_cosine(np.dot(n1, n2))
