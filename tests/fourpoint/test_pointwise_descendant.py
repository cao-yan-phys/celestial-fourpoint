import numpy as np

from celestial_intertwiners.fourpoint.pointwise_descendant import (
    EXPLICIT_ASTROMETRY_TO_PTA_NORMALIZATION,
    explicit_astrometry_antenna,
    explicit_pta_antenna,
    pointwise_descendant_residual,
    random_pointwise_descendant_check,
)


def test_axis_pointwise_descendant_normalization():
    omega = np.array([0.0, 0.0, 1.0])
    theta = 1.1
    phi = 0.7
    assert EXPLICIT_ASTROMETRY_TO_PTA_NORMALIZATION < 0
    for epsilon in (-1, 1):
        descendant, pta, error = pointwise_descendant_residual(theta, phi, omega, epsilon)
        assert abs(descendant - pta) / max(abs(pta), 1e-14) < 1e-9
        assert abs(explicit_astrometry_antenna(theta, phi, omega, epsilon)) > 0
        assert abs(explicit_pta_antenna(theta, phi, omega, epsilon)) > 0
        assert error < 1e-9


def test_random_pointwise_descendant_check():
    passed, max_error = random_pointwise_descendant_check(seed=123, trials=32, tolerance=1e-8)
    assert passed
    assert max_error < 1e-8
