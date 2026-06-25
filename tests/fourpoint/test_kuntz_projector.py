from celestial_intertwiners.fourpoint.kuntz_projector import (
    benchmark_mixed_projector_convergence,
    benchmark_mixed_spectral_convergence,
)


def test_single_leg_kuntz_master_projector_smoke_matches_direct():
    result = benchmark_mixed_projector_convergence(
        ("P", "P", "A", "P"),
        lmax_values=(4,),
        sample_power=8,
        direct_n_theta=32,
        direct_n_phi=64,
        backend="numpy",
    )

    entry = result.entries[0]
    assert entry.finite_fraction == 1.0
    assert entry.relative_error < 0.2


def test_reliable_spectral_convergence_entry_for_aaaa():
    result = benchmark_mixed_spectral_convergence(
        ("A", "A", "A", "A"),
        lmax_values=(4,),
        direct_n_theta=32,
        direct_n_phi=64,
        backend="numpy",
    )

    entry = result.entries[0]
    assert entry.lmax == 4
    assert entry.relative_error < 0.2
