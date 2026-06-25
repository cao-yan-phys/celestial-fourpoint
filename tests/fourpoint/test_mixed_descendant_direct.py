from celestial_intertwiners.fourpoint.block_benchmark import (
    benchmark_mixed_descendant_vs_direct,
    mixed_descendant_block_average,
    mixed_truncated_direct_average,
    verify_aaaa_descendant_vs_direct,
    verify_aaap_descendant_vs_direct,
    verify_aapp_descendant_vs_direct,
    verify_appp_descendant_vs_direct,
)
from celestial_intertwiners.fourpoint.kuntz_benchmark import (
    KUNTZ_SEED_EPSILONS,
    load_kuntz_fixtures,
)


def test_appp_descendant_matches_direct_quadrature():
    result = benchmark_mixed_descendant_vs_direct(
        ("A", "P", "P", "P"),
        lmax=3,
        tolerance=1e-11,
    )
    assert result.passed
    assert result.relative_error < 1e-11
    assert abs(result.direct) > 0
    assert verify_appp_descendant_vs_direct(lmax=3)


def test_aapp_descendant_matches_direct_quadrature():
    result = benchmark_mixed_descendant_vs_direct(
        ("A", "A", "P", "P"),
        lmax=3,
        tolerance=1e-11,
    )
    assert result.passed
    assert result.relative_error < 1e-11
    assert abs(result.direct) > 0
    assert verify_aapp_descendant_vs_direct(lmax=3)


def test_aaap_descendant_matches_direct_quadrature():
    result = benchmark_mixed_descendant_vs_direct(
        ("A", "A", "A", "P"),
        lmax=3,
        tolerance=1e-11,
    )
    assert result.passed
    assert result.relative_error < 1e-11
    assert abs(result.direct) > 0
    assert verify_aaap_descendant_vs_direct(lmax=3)


def test_aaaa_descendant_matches_direct_quadrature():
    result = benchmark_mixed_descendant_vs_direct(
        ("A", "A", "A", "A"),
        lmax=3,
        tolerance=1e-11,
    )
    assert result.passed
    assert result.relative_error < 1e-11
    assert abs(result.direct) > 0
    assert verify_aaaa_descendant_vs_direct(lmax=3)


def test_mixed_descendant_entry_points_on_kuntz_fixture():
    fixture = load_kuntz_fixtures()[0]
    observables = ("A", "P", "A", "P")
    descendant = mixed_descendant_block_average(
        observables,
        fixture.vectors,
        KUNTZ_SEED_EPSILONS,
        lmax=2,
    )
    direct = mixed_truncated_direct_average(
        observables,
        fixture.vectors,
        KUNTZ_SEED_EPSILONS,
        lmax=2,
        n_theta=12,
        n_phi=24,
    )
    assert abs(descendant - direct) / abs(direct) < 1e-11
