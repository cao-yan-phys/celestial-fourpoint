import numpy as np

from celestial_intertwiners.fourpoint.pointwise_descendant import helicity_polarization_from_axis
from celestial_intertwiners.fourpoint.tensor_reconstruction import (
    audit_helicity_reconstruction,
    bitensor_from_helicity_components,
    helicity_components_from_bitensor,
    helicity_components_from_real_vector,
    helicity_components_from_tangent_vector,
    helicity_to_real_vector,
    linear_polarizations_from_helicity,
    tangent_vector_from_components,
    tangent_vector_from_helicity_components,
    verify_aapp_real_bitensor_reconstruction,
    verify_appp_real_vector_reconstruction,
)


def test_helicity_real_vector_component_roundtrip():
    theta_component = 0.31
    phi_component = -0.47
    a_plus, a_minus = helicity_components_from_real_vector(theta_component, phi_component)
    reconstructed = helicity_to_real_vector(a_plus, a_minus)
    assert np.allclose(reconstructed, [theta_component, phi_component], atol=1e-14)


def test_cartesian_vector_and_bitensor_roundtrip():
    direction_a = np.array([0.2, -0.4, 0.8944271909999159])
    direction_b = np.array([-0.3, 0.5, 0.812403840463596])
    vector = tangent_vector_from_components(direction_a, 0.7, -0.2)
    a_plus, a_minus = helicity_components_from_tangent_vector(direction_a, vector)
    assert np.allclose(
        tangent_vector_from_helicity_components(direction_a, a_plus, a_minus),
        vector,
        atol=1e-14,
    )

    vector_b = tangent_vector_from_components(direction_b, -0.1, 0.9)
    bitensor = np.outer(vector, vector_b)
    components = helicity_components_from_bitensor(direction_a, direction_b, bitensor)
    assert np.allclose(
        bitensor_from_helicity_components(direction_a, direction_b, components),
        bitensor,
        atol=1e-14,
    )


def test_linear_polarizations_reconstruct_circular_basis():
    axis = np.array([0.4, -0.3, 0.8660254037844386])
    e_plus, e_cross = linear_polarizations_from_helicity(axis)
    assert np.allclose(e_plus + 1j * e_cross, helicity_polarization_from_axis(axis, 1), atol=1e-14)
    assert np.allclose(e_plus - 1j * e_cross, helicity_polarization_from_axis(axis, -1), atol=1e-14)


def test_physical_helicity_reconstruction_audit():
    result = audit_helicity_reconstruction(seed=1234, trials=24, tolerance=1e-11)
    assert result.passed
    assert result.max_polarization_error < 1e-11
    assert result.max_pta_error < 1e-11
    assert result.max_vector_error < 1e-11
    assert result.max_bitensor_error < 1e-11
    assert verify_appp_real_vector_reconstruction(seed=1234, trials=24)
    assert verify_aapp_real_bitensor_reconstruction(seed=1234, trials=24)
