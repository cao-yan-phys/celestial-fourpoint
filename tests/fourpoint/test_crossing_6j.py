import sympy as sp

from celestial_intertwiners.fourpoint.crossing import (
    recoupling_12_34_to_13_24,
    recoupling_inner_product_12_34_to_13_24,
    verify_wigner_6j_recoupling,
    wigner_6j,
)


def test_wigner_6j_symmetry_and_recoupling_helper():
    value = wigner_6j(2, 2, 2, 2, 2, 2)
    assert value == wigner_6j(2, 2, 2, 2, 2, 2)
    coeff = recoupling_12_34_to_13_24(2, 2, 2, 2, 2, 2)
    assert sp.simplify(coeff - value) == 0
    assert verify_wigner_6j_recoupling(2, 2, 2, 2, 2, 2)


def test_wigner_6j_phase_calibration_nontrivial_l():
    direct = recoupling_inner_product_12_34_to_13_24(1, 1, 2, 2, 1, 2)
    formula = recoupling_12_34_to_13_24(1, 1, 2, 2, 1, 2)
    assert sp.simplify(direct - formula) == 0
    assert verify_wigner_6j_recoupling(1, 1, 2, 2, 1, 2)
