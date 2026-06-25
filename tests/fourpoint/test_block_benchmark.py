from celestial_intertwiners.fourpoint.block_benchmark import (
    benchmark_pppp_block_vs_direct,
    pppp_coupled_block_average,
    pppp_truncated_direct_average,
    truncated_pta_response,
    verify_finite_block_higher_cutoff,
    verify_mixed_kernel_permutation_symmetry,
    verify_pppp_block_vs_direct,
    verify_pppp_permutation_symmetry,
)
from celestial_intertwiners.fourpoint.kuntz_benchmark import (
    KUNTZ_SEED_EPSILONS,
    load_kuntz_fixtures,
)


def test_truncated_pta_response_nonzero_on_kuntz_fixture():
    fixture = load_kuntz_fixtures()[0]
    value = truncated_pta_response(
        fixture.vectors[0],
        fixture.vectors[1],
        KUNTZ_SEED_EPSILONS[0],
        lmax=3,
    )
    assert abs(value) > 0


def test_pppp_finite_block_matches_direct_quadrature():
    result = benchmark_pppp_block_vs_direct(lmax=3, tolerance=1e-11)
    assert result.passed
    assert result.relative_error < 1e-11


def test_pppp_finite_block_and_direct_entry_points():
    fixture = load_kuntz_fixtures()[0]
    block = pppp_coupled_block_average(fixture.vectors, KUNTZ_SEED_EPSILONS, lmax=2)
    direct = pppp_truncated_direct_average(
        fixture.vectors,
        KUNTZ_SEED_EPSILONS,
        lmax=2,
        n_theta=12,
        n_phi=24,
    )
    assert abs(block - direct) / abs(direct) < 1e-11
    assert verify_pppp_block_vs_direct(lmax=3)


def test_finite_block_higher_cutoff_and_permutations():
    assert verify_finite_block_higher_cutoff()
    assert verify_pppp_permutation_symmetry()
    assert verify_mixed_kernel_permutation_symmetry()
