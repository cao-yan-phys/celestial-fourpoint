"""Direct source-direction quadrature scaffolding."""

from __future__ import annotations

import cmath
import math

import numpy as np
from numpy.polynomial.legendre import leggauss

try:
    from scipy.special import sph_harm_y as _scipy_sph_harm_y
except ImportError:
    from scipy.special import sph_harm as _scipy_sph_harm

    def sph_harm_y(l, m, theta, phi):
        return _scipy_sph_harm(m, l, phi, theta)

else:

    def sph_harm_y(l, m, theta, phi):
        return _scipy_sph_harm_y(l, m, theta, phi)


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Return approximately uniform unit vectors on the sphere."""

    if n_points <= 0:
        raise ValueError("n_points must be positive")
    points = []
    golden = np.pi * (3 - np.sqrt(5))
    for i in range(n_points):
        z = 1 - 2 * (i + 0.5) / n_points
        radius = np.sqrt(max(0.0, 1 - z * z))
        phi = i * golden
        points.append([radius * np.cos(phi), radius * np.sin(phi), z])
    return np.asarray(points)


def sphere_average(function, n_points: int = 2048):
    """Average a callable over source direction using Fibonacci quadrature."""

    values = [function(point) for point in fibonacci_sphere(n_points)]
    return sum(values) / n_points


def wigner_small_d(l: int, mp: int, m: int, theta: float) -> float:
    """Return Wigner small-d element d^l_{mp,m}(theta).

    The finite-sum convention is calibrated so that ``spin_weighted_Y`` with
    spin zero agrees with scipy's Condon-Shortley scalar spherical harmonics.
    """

    if abs(mp) > l or abs(m) > l:
        return 0.0
    log_prefactor = 0.5 * (
        math.lgamma(l + m + 1)
        + math.lgamma(l - m + 1)
        + math.lgamma(l + mp + 1)
        + math.lgamma(l - mp + 1)
    )
    total = 0.0
    cos_half = math.cos(theta / 2)
    sin_half = math.sin(theta / 2)
    for k in range(0, 2 * l + 1):
        denominator_args = (l + m - k, k, mp - m + k, l - mp - k)
        if any(arg < 0 for arg in denominator_args):
            continue
        cos_power = 2 * l + m - mp - 2 * k
        sin_power = mp - m + 2 * k
        if (cos_half == 0.0 and cos_power > 0) or (sin_half == 0.0 and sin_power > 0):
            continue
        log_denominator = sum(math.lgamma(arg + 1) for arg in denominator_args)
        log_term = log_prefactor - log_denominator
        if cos_power:
            log_term += cos_power * math.log(abs(cos_half))
        if sin_power:
            log_term += sin_power * math.log(abs(sin_half))
        sign = (-1) ** (k - m + mp)
        if cos_half < 0 and cos_power % 2:
            sign *= -1
        if sin_half < 0 and sin_power % 2:
            sign *= -1
        total += (
            sign
            * math.exp(log_term)
        )
    return total


def spin_weighted_spherical_harmonic(
    l: int,
    m: int,
    s: int,
    theta: float,
    phi: float,
) -> complex:
    """Return Goldberg-convention spin-weighted spherical harmonic."""

    if l < 0 or abs(m) > l or abs(s) > l:
        return 0.0j
    if s == 0:
        return sph_harm_y(l, m, theta, phi)
    return (
        (-1) ** s
        * math.sqrt((2 * l + 1) / (4 * math.pi))
        * wigner_small_d(l, m, -s, theta)
        * cmath.exp(1j * m * phi)
    )


def gauss_legendre_sphere_integral(function, *, n_theta: int = 80, n_phi: int = 160):
    """Integrate a function over the unit sphere."""

    xs, weights = leggauss(n_theta)
    phis = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    total = 0.0j
    for x, weight in zip(xs, weights):
        theta = math.acos(float(x))
        phi_sum = sum(function(theta, float(phi)) for phi in phis)
        total += weight * (2 * np.pi / n_phi) * phi_sum
    return total


def four_spin_integral_quadrature(
    ells: tuple[int, int, int, int],
    ms: tuple[int, int, int, int],
    spins: tuple[int, int, int, int],
    *,
    n_theta: int = 80,
    n_phi: int = 160,
) -> complex:
    """Numerically integrate a product of four spin-weighted harmonics."""

    def integrand(theta: float, phi: float):
        value = 1.0 + 0.0j
        for ell, m, spin in zip(ells, ms, spins):
            value *= spin_weighted_spherical_harmonic(ell, m, spin, theta, phi)
        return value

    return gauss_legendre_sphere_integral(integrand, n_theta=n_theta, n_phi=n_phi)


def verify_four_spin_integral_quadrature(
    exact_value,
    ells: tuple[int, int, int, int],
    ms: tuple[int, int, int, int],
    spins: tuple[int, int, int, int],
    *,
    tolerance: float = 1e-10,
    n_theta: int = 96,
    n_phi: int = 192,
) -> bool:
    """Compare direct quadrature with an exact four-spin value."""

    numeric = four_spin_integral_quadrature(
        ells, ms, spins, n_theta=n_theta, n_phi=n_phi
    )
    target = complex(exact_value.evalf()) if hasattr(exact_value, "evalf") else complex(exact_value)
    scale = max(abs(target), 1.0)
    return abs(numeric - target) / scale < tolerance
