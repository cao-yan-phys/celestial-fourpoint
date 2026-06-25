import sympy as sp

from celestial_intertwiners.fourpoint.conventions import (
    angular_output_spin,
    temporal_transfer,
)
from celestial_intertwiners.fourpoint.helicity_responses import verify_transfer_normalization
from celestial_intertwiners.fourpoint.response_expansions import (
    bispin_response_term,
    descendant_coefficient_residual,
    verify_bispin_descendant_coefficients,
)


def test_bispin_response_metadata():
    p_term = bispin_response_term("P", 1, 2)
    a_term = bispin_response_term("A", -1, 2)
    assert p_term.external_spin == 0
    assert a_term.external_spin == 1
    assert p_term.source_spin == 2
    assert a_term.source_spin == -2
    assert angular_output_spin("A", 1) == -1


def test_transfer_normalization_consistent():
    for ell in range(2, 14):
        assert verify_transfer_normalization(ell)
        for epsilon in (-1, 1):
            assert descendant_coefficient_residual(epsilon, ell) == 0
    assert verify_bispin_descendant_coefficients(14)


def test_temporal_transfer_separation():
    f = sp.Symbol("f")
    assert temporal_transfer("Z", f, exact=True) == 1
    assert temporal_transfer("A", f, exact=True) == 1
    assert temporal_transfer("P", f, exact=True) == 1 / (2 * sp.pi * sp.I * f)
    assert temporal_transfer("ADOT", f, exact=True) == 2 * sp.pi * sp.I * f
