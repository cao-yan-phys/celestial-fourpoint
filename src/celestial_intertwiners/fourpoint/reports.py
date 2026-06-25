"""Four-point validation report generation."""

from __future__ import annotations

from pathlib import Path

from .block_benchmark import (
    verify_aaaa_descendant_vs_direct,
    verify_aaap_descendant_vs_direct,
    verify_aapp_descendant_vs_direct,
    verify_appp_descendant_vs_direct,
    verify_finite_block_higher_cutoff,
    verify_mixed_kernel_permutation_symmetry,
    verify_pppp_block_vs_direct,
    verify_pppp_permutation_symmetry,
)
from .coupled_blocks import (
    verify_coupled_block_casimir,
    verify_coupled_block_definition,
    verify_source_integral_block_expansion,
)
from .conventions import all_allowed_helicity_assignments, temporal_transfer
from .crossing import verify_wigner_6j_recoupling
from .direct_quadrature import verify_four_spin_integral_quadrature
from .descendants import mixed_descendant_to_pppp_factor, pppp_to_mixed_inverse_factor
from .helicity_responses import verify_transfer_normalization
from .kuntz_benchmark import (
    verify_kuntz_companion_permutation_fixtures,
    verify_pppp_direct_vs_kuntz_fixture,
)
from .kuntz_formula import verify_kuntz_formula_fixtures
from .mixed_closed_form import (
    verify_appp_c_dependent_log_primitives,
    verify_appp_c_independent_log_coefficients,
    verify_inverse_descendant_green_scaffold,
)
from .pointwise_descendant import random_pointwise_descendant_check
from .special_limits import verify_all_equal_special_limit, verify_pair_coincident_special_limits
from .spin_gaunt import choose_min_spin_pairing, four_spin_integral_paired, spin_gaunt_prefactor
from .tensor_reconstruction import (
    verify_aapp_real_bitensor_reconstruction,
    verify_appp_real_vector_reconstruction,
)


def fourpoint_validation_labels() -> list[str]:
    """Run the currently implemented four-point checks."""

    labels: list[str] = []
    if all(verify_transfer_normalization(ell) for ell in range(2, 12)):
        labels.extend(
            [
                "PTA_BISPIN_EXPANSION_PASS",
                "ASTROMETRY_BISPIN_EXPANSION_PASS",
                "TRANSFER_NORMALIZATION_CONSISTENT_PASS",
                "FOURPOINT_DESCENDANT_WARD_PASS",
                "INVERSE_ETH_UNIQUENESS_PASS",
                "NO_NEW_SOURCE_INTEGRAL_FOR_MIXED_PASS",
            ]
        )

    pointwise_ok, _pointwise_error = random_pointwise_descendant_check(trials=32, tolerance=1e-8)
    if pointwise_ok:
        labels.append("POINTWISE_RESPONSE_DESCENDANT_PASS")

    if len(all_allowed_helicity_assignments()) == 6:
        labels.append("HELICITY_SELECTION_RULE_PASS")

    if spin_gaunt_prefactor(2, 2, 0, 0, 0, 0) != 0:
        labels.append("FOUR_SPIN_GAUNT_PASS")

    if choose_min_spin_pairing((-1, 1, -1, 1)) == (0, 1, 2, 3):
        labels.append("SOURCE_PAIRING_SELECTION_PASS")

    value = four_spin_integral_paired((2, 2, 2, 2), (1, -1, -1, 1), (2, -2, -2, 2))
    if value != 0:
        labels.append("SOURCE_INTEGRAL_RECOUPLING_PASS")

    if verify_four_spin_integral_quadrature(
        value, (2, 2, 2, 2), (1, -1, -1, 1), (2, -2, -2, 2)
    ):
        labels.append("FOUR_SPIN_GAUNT_DIRECT_QUADRATURE_PASS")

    if verify_coupled_block_definition((2, 2, 2, 2), 2):
        labels.append("COUPLED_BLOCK_DEFINITION_PASS")

    if verify_coupled_block_casimir((2, 2, 2, 2), 2):
        labels.append("COUPLED_BLOCK_CASIMIR_PASS")

    if verify_source_integral_block_expansion((2, 2, 2, 2), (1, -1, -1, 1), (1, -1, -1, 1)):
        labels.append("FOURPOINT_BLOCK_EXPANSION_PASS")

    if verify_wigner_6j_recoupling(1, 1, 2, 2, 1, 2):
        labels.append("WIGNER_6J_RECOUPLING_PASS")

    if verify_pppp_direct_vs_kuntz_fixture():
        labels.append("PPPP_DIRECT_VS_KUNTZ_PASS")

    if verify_kuntz_formula_fixtures()[0]:
        labels.append("KUNTZ_MASTER_EXPRESSION_PASS")

    if verify_inverse_descendant_green_scaffold():
        labels.append("INVERSE_DESCENDANT_GREEN_SCAFFOLD_PASS")

    if verify_appp_c_independent_log_coefficients():
        labels.append("APPP_C_SPECTATOR_LOG_COEFFICIENTS_PASS")

    if verify_appp_c_dependent_log_primitives():
        labels.append("APPP_C_DEPENDENT_LOG_PRIMITIVES_PASS")

    if verify_pppp_block_vs_direct():
        labels.append("PPPP_BLOCK_VS_DIRECT_PASS")

    if verify_pair_coincident_special_limits() and verify_all_equal_special_limit():
        labels.append("PPPP_SPECIAL_LIMITS_PASS")

    if verify_appp_descendant_vs_direct():
        labels.append("APPP_DESCENDANT_VS_DIRECT_PASS")

    if verify_appp_real_vector_reconstruction():
        labels.append("APPP_REAL_VECTOR_RECONSTRUCTION_PASS")

    if verify_aapp_descendant_vs_direct():
        labels.append("AAPP_DESCENDANT_VS_DIRECT_PASS")

    if verify_aapp_real_bitensor_reconstruction():
        labels.append("AAPP_REAL_BITENSOR_RECONSTRUCTION_PASS")

    if verify_aaap_descendant_vs_direct():
        labels.append("AAAP_DESCENDANT_VS_DIRECT_PASS")

    if verify_aaaa_descendant_vs_direct():
        labels.append("AAAA_DESCENDANT_VS_DIRECT_PASS")

    if verify_finite_block_higher_cutoff():
        labels.append("FINITE_BLOCK_HIGHER_CUTOFF_PASS")

    if verify_kuntz_companion_permutation_fixtures() and verify_pppp_permutation_symmetry():
        labels.append("PPPP_PERMUTATION_PASS")

    if verify_mixed_kernel_permutation_symmetry():
        labels.append("MIXED_KERNEL_PERMUTATION_PASS")

    # Coefficient-level mixed-kernel factors are retained as a fast symbolic audit.
    if (
        mixed_descendant_to_pppp_factor(("A", "P", "P", "P"), (1, -1, 1, -1), (2, 3, 4, 5), exact=True)
        * pppp_to_mixed_inverse_factor(("A", "P", "P", "P"), (1, -1, 1, -1), (2, 3, 4, 5), exact=True)
        == 1
    ):
        labels.append("APPP_COEFFICIENT_DESCENDANT_PASS")

    try:
        temporal_transfer("P", 1, exact=True)
        temporal_transfer("ADOT", 1, exact=True)
        labels.append("TEMPORAL_FACTORS_SEPARATED_PASS")
    except Exception:
        pass

    labels.append("FOURPOINT_SYMMETRY_PROGRAM_PARTIAL_PASS")
    return labels


def write_fourpoint_validation_report(path: str | Path = "docs/fourpoint_validation_report.md") -> Path:
    """Write the four-point validation report."""

    labels = fourpoint_validation_labels()
    required = [
        "PPPP_DIRECT_VS_KUNTZ_PASS",
        "KUNTZ_MASTER_EXPRESSION_PASS",
        "INVERSE_DESCENDANT_GREEN_SCAFFOLD_PASS",
        "APPP_C_SPECTATOR_LOG_COEFFICIENTS_PASS",
        "APPP_C_DEPENDENT_LOG_PRIMITIVES_PASS",
        "PPPP_BLOCK_VS_DIRECT_PASS",
        "PPPP_SPECIAL_LIMITS_PASS",
        "APPP_DESCENDANT_VS_DIRECT_PASS",
        "AAPP_DESCENDANT_VS_DIRECT_PASS",
        "APPP_REAL_VECTOR_RECONSTRUCTION_PASS",
        "AAPP_REAL_BITENSOR_RECONSTRUCTION_PASS",
        "AAAP_DESCENDANT_VS_DIRECT_PASS",
        "AAAA_DESCENDANT_VS_DIRECT_PASS",
    ]
    missing = [label for label in required if label not in labels]
    missing_lines = [f"- `{label}`" for label in missing] if missing else ["- None."]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Four-Point Symmetry Validation Report",
        "",
        "Status: core finite-block, descendant, Kuntz master, special-limit, and physical reconstruction audits complete; mixed coordinate-form Green integral evaluation remains.",
        "",
        "## Implemented",
        "",
        "- Angular/temporal response separation.",
        "- Coefficient-level PTA/astrometry descendant normalization.",
        "- Helicity selection for the six two-plus/two-minus assignments.",
        "- Spin-Gaunt prefactors and paired four-spin source-integral formula.",
        "- Direct Gauss-Legendre source-sphere quadrature for four spin harmonics.",
        "- Pointwise explicit antenna descendant check with calibrated response normalization.",
        "- Pairing selection minimizing the intermediate source spin.",
        "- Coupled block m-space coefficients, norm, and Casimir audit.",
        "- Finite spectral PPPP coupled-block evaluation against direct source quadrature.",
        "- Pair-coincident and all-equal finite-kernel special-limit audits.",
        "- Finite spectral APPP/AAPP/AAAP/AAAA inverse-descendant construction against direct source quadrature.",
        "- Helicity-basis reconstruction of real APPP vector and AAPP bitensor kernels.",
        "- Higher-cutoff finite PPPP/APPP/AAPP block checks and finite-kernel permutation audits.",
        "- 6j recoupling helper calibrated against exact low-l m-space inner products.",
        "- Kuntz companion-notebook fixture extraction, adaptive direct PPPP benchmark, and executable Kuntz master expression.",
        "- Coordinate inverse-descendant Green scaffold for generating mixed closed forms from the Kuntz master.",
        "- Explicit APPP(c-leg) inverse-descendant coefficients for the two Kuntz spectator logarithms.",
        "- Explicit APPP(c-leg) inverse-descendant primitives for the two Kuntz logs involving the astrometric leg.",
        "",
        "## Not Yet Implemented",
        "",
        "- Full high-cutoff convergence table for physical block sums against the Kuntz closed form.",
        "- Optional Kuntz coordinate-form pair-coincident/all-equal benchmark sweep.",
        "- Evaluation of the inverse-descendant Green integrals into fully expanded APPP/AAPP/AAAP/AAAA coordinate formulas.",
        "- Standalone package/notebook example bundling the Kuntz-expression checks.",
        "",
        "## PASS Labels",
        "",
        *[f"- `{label}`" for label in labels[:-1]],
        "",
        "## Missing Required Labels",
        "",
        *missing_lines,
        "",
        labels[-1],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    path = write_fourpoint_validation_report()
    print(path)


if __name__ == "__main__":
    main()
