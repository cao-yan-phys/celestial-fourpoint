import sympy as sp

from celestial_intertwiners.fourpoint.coupled_blocks import (
    casimir_eigenvalue,
    coupled_block_m_coefficient,
    coupled_block_norm,
    external_spins,
    intermediate_L_values,
    source_recoupling_coefficient,
    source_integral_block_sum,
    transfer_product,
    verify_coupled_block_casimir,
    verify_coupled_block_definition,
    verify_source_integral_block_expansion,
)
from celestial_intertwiners.fourpoint.spin_gaunt import four_spin_integral_paired


def test_coupled_block_bookkeeping():
    ells = (2, 2, 2, 2)
    assert intermediate_L_values(ells) == (0, 1, 2, 3, 4)
    assert casimir_eigenvalue(3) == 12
    assert external_spins(("A", "P", "A", "P"), (1, -1, -1, 1)) == (-1, 0, 1, 0)
    assert verify_coupled_block_definition(ells, 2)
    assert verify_coupled_block_casimir(ells, 2)
    assert coupled_block_norm(ells, 2) != 0
    assert coupled_block_m_coefficient(ells, 2, (1, -1, -1, 1)) != 0


def test_source_recoupling_and_transfer_product():
    ells = (2, 2, 2, 2)
    eps = (-1, 1, -1, 1)
    coeff = source_recoupling_coefficient(ells, eps, 0, pairing=(0, 1, 2, 3))
    assert coeff != 0
    assert source_recoupling_coefficient(ells, (1, 1, 1, -1), 0) == 0
    assert transfer_product(("P", "A", "P", "A"), ells, exact=True) != 0
    assert sp.simplify(coeff - coeff) == 0


def test_source_integral_block_expansion():
    ells = (2, 2, 2, 2)
    ms = (1, -1, -1, 1)
    eps = (1, -1, -1, 1)
    direct = four_spin_integral_paired(ells, ms, tuple(2 * e for e in eps))
    block = source_integral_block_sum(ells, ms, eps)
    assert sp.simplify(block - direct) == 0
    assert verify_source_integral_block_expansion(ells, ms, eps)
