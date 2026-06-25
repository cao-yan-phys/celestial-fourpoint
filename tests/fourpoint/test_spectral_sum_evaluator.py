import pytest

from celestial_intertwiners.fourpoint.kuntz_benchmark import KUNTZ_SEED_EPSILONS, load_kuntz_fixtures
from celestial_intertwiners.fourpoint.spectral_convergence import spectral_truncated_direct_average
from celestial_intertwiners.fourpoint.spectral_sum_evaluator import SpectralSumEvaluator


def test_spectral_sum_evaluator_matches_reference_lsum():
    fixture = load_kuntz_fixtures()[0]
    evaluator = SpectralSumEvaluator(lmax=5, n_theta=18, n_phi=36, backend="numpy")
    for observables in (
        ("P", "P", "P", "P"),
        ("A", "P", "P", "P"),
        ("A", "A", "P", "P"),
    ):
        fast = evaluator.evaluate(observables, fixture.vectors, KUNTZ_SEED_EPSILONS).value
        reference = spectral_truncated_direct_average(
            observables,
            fixture.vectors,
            KUNTZ_SEED_EPSILONS,
            lmax=5,
            n_theta=18,
            n_phi=36,
        )
        assert abs(fast - reference) < 1e-13


def test_spectral_sum_evaluator_cuda_optional_matches_cpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device available")
    fixture = load_kuntz_fixtures()[0]
    cpu = SpectralSumEvaluator(lmax=5, n_theta=18, n_phi=36, backend="numpy")
    gpu = SpectralSumEvaluator(lmax=5, n_theta=18, n_phi=36, backend="cuda")
    observables = ("A", "A", "P", "P")
    cpu_value = cpu.evaluate(observables, fixture.vectors, KUNTZ_SEED_EPSILONS).value
    gpu_value = gpu.evaluate(observables, fixture.vectors, KUNTZ_SEED_EPSILONS).value
    assert abs(cpu_value - gpu_value) < 1e-12
