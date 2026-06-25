"""Spin-coupled four-point kernels for PTA and astrometry."""

from .conventions import (
    all_allowed_helicity_assignments,
    angular_output_spin,
    helicity_selection_rule,
    temporal_transfer,
)
from .descendants import descendant_factor, inverse_descendant_factor
from .fast_evaluator import FastFourPointEvaluator, fast_physical_mixed_average
from .helicity_responses import response_transfer, verify_transfer_normalization
from .kuntz_formula import (
    evaluate_kuntz_pppp_closed_form,
    evaluate_kuntz_pppp_closed_form_angles,
    kuntz_pppp_sympy_expression,
)
from .kuntz_projector import (
    benchmark_mixed_projector_convergence,
    benchmark_mixed_spectral_convergence,
    evaluate_mixed_projector_from_kuntz_master_angles,
    evaluate_mixed_spectral_kernel,
)
from .precomputed_spectral_kernel import (
    PrecomputedSpectralKernel,
    build_precomputed_spectral_kernel,
    default_precomputed_kernel_path,
    load_precomputed_spectral_kernel,
)
from .mixed_closed_form import (
    appp_c_leg_full_primitive_u,
    appp_c_leg_log_correction_primitives,
    appp_c_leg_dependent_log_primitives,
    appp_c_leg_independent_log_coefficients,
    appp_c_leg_rational_residue_primitives,
    appp_c_descendant_closure_residual,
    benchmark_appp_c_projected_against_direct,
    benchmark_appp_c_representative_against_direct,
    evaluate_appp_c_projected_from_kuntz_master_angles,
    evaluate_appp_c_representative_closed_form_angles,
    coordinate_descendant_operator,
    inverse_descendant_mode_formula,
    inverse_descendant_mode_residual,
    kuntz_canonical_vectors_from_angles,
)
from .spectral_sum_evaluator import SpectralSumEvaluator, spectral_sum_average

__all__ = [
    "all_allowed_helicity_assignments",
    "angular_output_spin",
    "helicity_selection_rule",
    "temporal_transfer",
    "descendant_factor",
    "inverse_descendant_factor",
    "FastFourPointEvaluator",
    "fast_physical_mixed_average",
    "evaluate_kuntz_pppp_closed_form",
    "evaluate_kuntz_pppp_closed_form_angles",
    "kuntz_pppp_sympy_expression",
    "evaluate_mixed_projector_from_kuntz_master_angles",
    "benchmark_mixed_projector_convergence",
    "evaluate_mixed_spectral_kernel",
    "benchmark_mixed_spectral_convergence",
    "PrecomputedSpectralKernel",
    "default_precomputed_kernel_path",
    "build_precomputed_spectral_kernel",
    "load_precomputed_spectral_kernel",
    "coordinate_descendant_operator",
    "inverse_descendant_mode_formula",
    "inverse_descendant_mode_residual",
    "appp_c_leg_independent_log_coefficients",
    "appp_c_leg_dependent_log_primitives",
    "appp_c_leg_rational_residue_primitives",
    "appp_c_leg_log_correction_primitives",
    "appp_c_leg_full_primitive_u",
    "evaluate_appp_c_representative_closed_form_angles",
    "evaluate_appp_c_projected_from_kuntz_master_angles",
    "appp_c_descendant_closure_residual",
    "benchmark_appp_c_representative_against_direct",
    "benchmark_appp_c_projected_against_direct",
    "kuntz_canonical_vectors_from_angles",
    "SpectralSumEvaluator",
    "spectral_sum_average",
    "response_transfer",
    "verify_transfer_normalization",
]
