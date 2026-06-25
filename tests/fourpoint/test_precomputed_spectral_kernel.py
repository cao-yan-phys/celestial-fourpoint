from pathlib import Path

from celestial_intertwiners.fourpoint.kuntz_benchmark import (
    KUNTZ_SEED_EPSILONS,
    load_kuntz_fixtures,
)
from celestial_intertwiners.fourpoint.precomputed_spectral_kernel import (
    PrecomputedSpectralKernel,
)
from celestial_intertwiners.fourpoint.spectral_sum_evaluator import SpectralSumEvaluator


def test_precomputed_spectral_kernel_roundtrip_matches_evaluator():
    fixture = load_kuntz_fixtures()[0]
    kernel = PrecomputedSpectralKernel.build(lmax=4, n_theta=16, n_phi=32)
    path = Path("output") / "test_artifacts" / "precomputed_test_kernel.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    info = kernel.save(path)
    loaded = PrecomputedSpectralKernel.load(path)

    assert info.mode_count > 0
    assert info.grid_size == 16 * 32

    observables = ("A", "A", "P", "P")
    precomputed = loaded.evaluate(observables, fixture.vectors, KUNTZ_SEED_EPSILONS).value
    reference = SpectralSumEvaluator(
        lmax=4,
        n_theta=16,
        n_phi=32,
        backend="numpy",
    ).evaluate(observables, fixture.vectors, KUNTZ_SEED_EPSILONS).value

    assert abs(precomputed - reference) < 1e-13


def test_precomputed_spectral_kernel_family_keys():
    fixture = load_kuntz_fixtures()[0]
    kernel = PrecomputedSpectralKernel.build(lmax=4, n_theta=16, n_phi=32)
    family = kernel.evaluate_family(fixture.vectors, KUNTZ_SEED_EPSILONS)

    assert set(family) == {
        ("P", "P", "P", "P"),
        ("A", "P", "P", "P"),
        ("A", "A", "P", "P"),
        ("A", "A", "A", "P"),
        ("A", "A", "A", "A"),
    }
