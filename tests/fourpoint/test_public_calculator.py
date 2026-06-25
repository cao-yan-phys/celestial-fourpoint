from celestial_fourpoint import FourPointCalculator, fixture_accuracy_sweep
from celestial_intertwiners.fourpoint.kuntz_benchmark import (
    KUNTZ_SEED_EPSILONS,
    load_kuntz_fixtures,
)


def test_public_calculator_uses_precomputed_kernel():
    fixture = load_kuntz_fixtures()[0]
    calculator = FourPointCalculator.load_precomputed(lmax=15)

    value = calculator.evaluate(("A", "A", "A", "A"), fixture.vectors, KUNTZ_SEED_EPSILONS)
    family = calculator.family(fixture.vectors, KUNTZ_SEED_EPSILONS)

    assert value == family[("A", "A", "A", "A")]
    assert abs(calculator.kuntz_pppp(fixture.vectors)) > 0


def test_public_accuracy_sweep_smoke():
    reports = fixture_accuracy_sweep(
        lmax=15,
        fixture_count=1,
        direct_n_theta=32,
        direct_n_phi=64,
        backend="numpy",
    )

    assert len(reports) == 1
    assert reports[0].fixture_name
    assert reports[0].max_cached_direct_relative_error < 0.1
