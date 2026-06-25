import sympy as sp

from celestial_intertwiners.fourpoint.conventions import (
    all_allowed_helicity_assignments,
    helicity_selection_rule,
)
from celestial_intertwiners.fourpoint.direct_quadrature import (
    four_spin_integral_quadrature,
    sph_harm_y,
    spin_weighted_spherical_harmonic,
    verify_four_spin_integral_quadrature,
)
from celestial_intertwiners.fourpoint.spin_gaunt import (
    choose_min_spin_pairing,
    four_spin_integral_paired,
    pairing_spin_cost,
    spin_gaunt_prefactor,
)


def test_helicity_selection_rule():
    allowed = all_allowed_helicity_assignments()
    assert len(allowed) == 6
    assert helicity_selection_rule((-1, 1, -1, 1))
    assert not helicity_selection_rule((1, 1, 1, -1))


def test_spin_gaunt_selection():
    assert spin_gaunt_prefactor(2, 2, 0, 0, 0, 1) == 0
    assert spin_gaunt_prefactor(2, 2, 0, 0, 0, 0) != 0


def test_pairing_selection_minimizes_intermediate_spin():
    eps = (-1, 1, -1, 1)
    pairing = choose_min_spin_pairing(eps)
    assert pairing_spin_cost(eps, pairing) == 0


def test_four_spin_integral_paired_nonzero_and_selection():
    value = four_spin_integral_paired((2, 2, 2, 2), (1, -1, -1, 1), (2, -2, -2, 2))
    assert value != 0
    assert four_spin_integral_paired((2, 2, 2, 2), (1, -1, -1, 1), (2, 2, -2, 2)) == 0
    assert sp.simplify(value - value) == 0


def test_spin_weighted_harmonic_scalar_limit():
    theta = 1.1
    phi = 0.7
    for ell in range(0, 4):
        for m in range(-ell, ell + 1):
            actual = spin_weighted_spherical_harmonic(ell, m, 0, theta, phi)
            expected = sph_harm_y(ell, m, theta, phi)
            assert abs(actual - expected) < 1e-14


def test_four_spin_integral_direct_quadrature():
    cases = [
        ((2, 2, 2, 2), (1, -1, -1, 1), (2, -2, -2, 2)),
        ((2, 3, 2, 3), (0, 1, 0, -1), (2, -2, 2, -2)),
        ((3, 3, 2, 2), (1, -1, 1, -1), (0, 0, 0, 0)),
    ]
    for ells, ms, spins in cases:
        exact = four_spin_integral_paired(ells, ms, spins)
        numeric = four_spin_integral_quadrature(ells, ms, spins, n_theta=96, n_phi=192)
        assert abs(numeric - complex(exact.evalf())) < 1e-10
        assert verify_four_spin_integral_quadrature(exact, ells, ms, spins)
