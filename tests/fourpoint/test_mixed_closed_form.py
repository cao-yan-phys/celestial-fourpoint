import sympy as sp

from celestial_intertwiners.fourpoint.mixed_closed_form import (
    CC,
    QC,
    PSIC,
    UC,
    ZD,
    appp_c_leg_full_primitive_u,
    appp_c_leg_log_correction_primitives,
    appp_c_leg_dependent_log_primitives,
    appp_c_leg_independent_log_coefficients,
    appp_c_leg_rational_residue_primitives,
    benchmark_appp_c_projected_against_direct,
    coordinate_descendant_operator,
    inverse_descendant_integrating_factor,
    inverse_descendant_mode_formula,
    inverse_descendant_mode_residual,
    kuntz_master_descendant_source,
    leg_coordinates,
    verify_appp_c_dependent_log_primitives,
    verify_appp_c_full_primitive_components,
    verify_appp_c_independent_log_coefficients,
    verify_appp_c_log_correction_primitives,
    verify_appp_c_rational_residue_primitives,
)


def test_coordinate_descendant_operator_matches_spin_convention():
    test_component = sp.sqrt(1 - CC**2)

    assert sp.simplify(coordinate_descendant_operator(test_component, "c", 1) + 2 * CC) == 0
    assert sp.simplify(coordinate_descendant_operator(test_component, "c", -1) + 2 * CC) == 0

    winding_component = test_component * sp.exp(sp.I * PSIC)
    descendant = coordinate_descendant_operator(winding_component, "c", 1)
    assert sp.simplify(descendant - (1 - 2 * CC) * sp.exp(sp.I * PSIC)) == 0


def test_inverse_descendant_fourier_mode_residual():
    candidate = sp.sqrt(1 - CC**2)
    source = -2 * CC

    assert (
        inverse_descendant_mode_residual(
            candidate,
            source,
            mode=0,
            epsilon=1,
            cosine=CC,
        )
        == 0
    )


def test_inverse_descendant_integrating_factor_solves_homogeneous_ode():
    mode = 2
    epsilon = -1
    factor = inverse_descendant_integrating_factor(mode=mode, epsilon=epsilon, cosine=CC)
    homogeneous_solution = 1 / factor

    assert (
        inverse_descendant_mode_residual(
            homogeneous_solution,
            0,
            mode=mode,
            epsilon=epsilon,
            cosine=CC,
        )
        == 0
    )


def test_inverse_descendant_mode_formula_keeps_constant_explicit():
    formula = inverse_descendant_mode_formula(-2 * CC, mode=0, epsilon=1, cosine=CC)

    assert formula.integrating_factor == sp.sqrt(1 - CC**2)
    assert formula.solution.has(sp.Integral)
    assert formula.solution.has(sp.Symbol("C_0"))


def test_kuntz_master_source_uses_c_and_d_leg_coordinates():
    source = kuntz_master_descendant_source()
    assert leg_coordinates("c").cosine in source.free_symbols
    assert leg_coordinates("d").azimuth in source.free_symbols


def test_appp_c_spectator_log_coefficients_are_explicitly_integrated():
    coefficients = appp_c_leg_independent_log_coefficients()
    assert [coefficient.log_argument for coefficient in coefficients] == [
        "2/(1 - cb)",
        "2/(1 - cd)",
    ]
    assert verify_appp_c_independent_log_coefficients()

    for coefficient in coefficients:
        assert coefficient.coefficient_characteristic.has(QC)
        assert coefficient.coefficient_characteristic.has(ZD)
        assert not coefficient.coefficient_coordinate.has(QC)
        assert not coefficient.coefficient_coordinate.has(ZD)


def test_appp_c_dependent_log_coefficients_have_explicit_primitives():
    primitives = appp_c_leg_dependent_log_primitives()
    assert len(primitives) == 2
    assert verify_appp_c_dependent_log_primitives()

    for primitive in primitives:
        assert primitive.source_u.has(UC)
        assert primitive.coefficient_u.has(UC)
        assert not primitive.primitive_u.has(sp.Integral)


def test_appp_c_rational_residue_has_explicit_primitives():
    primitives = appp_c_leg_rational_residue_primitives()
    assert [primitive.term_index for primitive in primitives] == [0, 1, 2, 3]
    assert verify_appp_c_rational_residue_primitives()

    for primitive in primitives:
        assert primitive.source_u.has(UC)
        assert primitive.coefficient_u.has(UC)
        assert not primitive.primitive_u.has(sp.Integral)


def test_appp_c_log_corrections_have_structured_primitives():
    primitives = appp_c_leg_log_correction_primitives()
    assert len(primitives) == 2
    assert verify_appp_c_log_correction_primitives()

    for primitive in primitives:
        assert primitive.source_u.has(UC)
        assert primitive.source_y.has(sp.Symbol("yc"))
        assert primitive.primitive_u.has(sp.log)
        assert not primitive.primitive_u.has(sp.Integral)


def test_appp_c_full_primitive_assembles_without_unevaluated_integrals():
    full = appp_c_leg_full_primitive_u()
    assert verify_appp_c_full_primitive_components()
    assert full.primitive_u.has(sp.log)
    assert full.response_u.has(QC)
    assert not full.primitive_u.has(sp.Integral)


def test_appp_c_projected_branch_matches_canonical_direct():
    audit = benchmark_appp_c_projected_against_direct(
        lmax=12,
        direct_n_theta=96,
        direct_n_phi=192,
    )

    assert audit.projected_relative_error < 5e-3
    assert audit.pppp_reconstruction_relative_error < 1e-2
    assert audit.representative_relative_error > 10
