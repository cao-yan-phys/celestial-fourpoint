import pytest

from celestial_intertwiners.fourpoint.fast_evaluator import FastFourPointEvaluator
from celestial_intertwiners.fourpoint.kuntz_benchmark import KUNTZ_SEED_EPSILONS, load_kuntz_fixtures
from celestial_intertwiners.fourpoint.spectral_convergence import explicit_physical_mixed_average


def test_fast_evaluator_matches_scalar_direct_grid():
    fixture = load_kuntz_fixtures()[0]
    evaluator = FastFourPointEvaluator(n_theta=18, n_phi=36, backend="numpy")
    cases = (
        ("P", "P", "P", "P"),
        ("A", "P", "P", "P"),
        ("A", "A", "P", "P"),
        ("A", "A", "A", "P"),
        ("A", "A", "A", "A"),
    )
    for observables in cases:
        fast = evaluator.evaluate(observables, fixture.vectors, KUNTZ_SEED_EPSILONS).value
        scalar = explicit_physical_mixed_average(
            observables,
            fixture.vectors,
            KUNTZ_SEED_EPSILONS,
            n_theta=18,
            n_phi=36,
        )
        assert abs(fast - scalar) < 1e-13


def test_fast_evaluator_family_keys():
    fixture = load_kuntz_fixtures()[0]
    evaluator = FastFourPointEvaluator(n_theta=12, n_phi=24, backend="numpy")
    family = evaluator.evaluate_family(fixture.vectors, KUNTZ_SEED_EPSILONS)
    assert set(family) == {
        ("P", "P", "P", "P"),
        ("A", "P", "P", "P"),
        ("A", "A", "P", "P"),
        ("A", "A", "A", "P"),
        ("A", "A", "A", "A"),
    }
    assert all(abs(value) > 0 for value in family.values())


def test_fast_evaluator_cuda_optional_matches_cpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device available")
    fixture = load_kuntz_fixtures()[0]
    cpu = FastFourPointEvaluator(n_theta=10, n_phi=20, backend="numpy")
    gpu = FastFourPointEvaluator(n_theta=10, n_phi=20, backend="cuda")
    observables = ("A", "A", "P", "P")
    cpu_value = cpu.evaluate(observables, fixture.vectors, KUNTZ_SEED_EPSILONS).value
    gpu_value = gpu.evaluate(observables, fixture.vectors, KUNTZ_SEED_EPSILONS).value
    assert abs(cpu_value - gpu_value) < 1e-12
