from celestial_intertwiners.fourpoint.kuntz_benchmark import KUNTZ_SEED_EPSILONS, load_kuntz_fixtures
from celestial_intertwiners.fourpoint.spectral_convergence import compare_physical_mixed_spectral


def test_physical_mixed_spectral_matches_explicit_direct():
    fixture = load_kuntz_fixtures()[0]
    cases = (
        (("A", "P", "P", "P"), 3.0e-2),
        (("A", "A", "P", "P"), 1.0e-2),
        (("A", "A", "A", "P"), 5.0e-3),
        (("A", "A", "A", "A"), 5.0e-3),
    )
    for observables, tolerance in cases:
        result = compare_physical_mixed_spectral(
            observables,
            fixture.vectors,
            KUNTZ_SEED_EPSILONS,
            lmax=10,
            direct_n_theta=56,
            direct_n_phi=112,
            tolerance=tolerance,
        )
        assert result.passed
