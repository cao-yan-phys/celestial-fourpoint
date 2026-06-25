from celestial_intertwiners.fourpoint.reports import fourpoint_validation_labels


def test_fourpoint_report_labels_partial():
    labels = fourpoint_validation_labels()
    assert "PTA_BISPIN_EXPANSION_PASS" in labels
    assert "ASTROMETRY_BISPIN_EXPANSION_PASS" in labels
    assert "POINTWISE_RESPONSE_DESCENDANT_PASS" in labels
    assert "FOUR_SPIN_GAUNT_PASS" in labels
    assert "FOUR_SPIN_GAUNT_DIRECT_QUADRATURE_PASS" in labels
    assert "SOURCE_PAIRING_SELECTION_PASS" in labels
    assert "COUPLED_BLOCK_DEFINITION_PASS" in labels
    assert "COUPLED_BLOCK_CASIMIR_PASS" in labels
    assert "FOURPOINT_BLOCK_EXPANSION_PASS" in labels
    assert "WIGNER_6J_RECOUPLING_PASS" in labels
    assert "PPPP_DIRECT_VS_KUNTZ_PASS" in labels
    assert "KUNTZ_MASTER_EXPRESSION_PASS" in labels
    assert "INVERSE_DESCENDANT_GREEN_SCAFFOLD_PASS" in labels
    assert "PPPP_BLOCK_VS_DIRECT_PASS" in labels
    assert "PPPP_SPECIAL_LIMITS_PASS" in labels
    assert "APPP_DESCENDANT_VS_DIRECT_PASS" in labels
    assert "APPP_REAL_VECTOR_RECONSTRUCTION_PASS" in labels
    assert "AAPP_DESCENDANT_VS_DIRECT_PASS" in labels
    assert "AAPP_REAL_BITENSOR_RECONSTRUCTION_PASS" in labels
    assert "AAAP_DESCENDANT_VS_DIRECT_PASS" in labels
    assert "AAAA_DESCENDANT_VS_DIRECT_PASS" in labels
    assert "FINITE_BLOCK_HIGHER_CUTOFF_PASS" in labels
    assert "PPPP_PERMUTATION_PASS" in labels
    assert "MIXED_KERNEL_PERMUTATION_PASS" in labels
    assert labels[-1] == "FOURPOINT_SYMMETRY_PROGRAM_PARTIAL_PASS"
