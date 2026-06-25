import numpy as np

from celestial_intertwiners.fourpoint.kuntz_benchmark import (
    KUNTZ_SEED_EPSILONS,
    benchmark_kuntz_fixture,
    evaluate_kuntz_pppp_seed,
    kuntz_closed_formula_available,
    load_kuntz_fixtures,
    pppp_direct_fibonacci_average,
    vec2angles,
    verify_kuntz_companion_permutation_fixtures,
)


def test_kuntz_fixture_loader_and_vec2angles():
    assert kuntz_closed_formula_available()
    fixtures = load_kuntz_fixtures()
    fixture = fixtures[0]
    assert fixture.name == "notebook_abcd"
    assert fixture.vectors.shape == (4, 3)
    assert evaluate_kuntz_pppp_seed(0) == fixture.value
    assert np.allclose(vec2angles(*fixture.vectors), fixture.angles, atol=1e-13)
    assert verify_kuntz_companion_permutation_fixtures()


def test_pppp_direct_adaptive_matches_kuntz_fixture():
    result = benchmark_kuntz_fixture(0, tolerance=1e-8)
    assert result.passed
    assert result.relative_error < 1e-8


def test_pppp_fibonacci_smoke_converges_toward_kuntz_fixture():
    fixture = load_kuntz_fixtures()[0]
    direct = pppp_direct_fibonacci_average(
        fixture.vectors,
        KUNTZ_SEED_EPSILONS,
        n_points=2048,
    )
    assert abs(direct - fixture.value) / abs(fixture.value) < 1e-2
