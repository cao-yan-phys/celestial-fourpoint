from celestial_intertwiners.fourpoint.special_limits import (
    audit_special_limits,
    verify_all_equal_special_limit,
    verify_pair_coincident_special_limits,
)


def test_pair_coincident_and_all_equal_special_limits_pass():
    audit = audit_special_limits(lmax=4, tolerance=1e-10, absolute_tolerance=1e-12)
    assert audit.passed
    assert audit.max_absolute_error < 1e-12
    assert verify_pair_coincident_special_limits(lmax=4)
    assert verify_all_equal_special_limit(lmax=4)


def test_all_equal_appp_zero_limit_uses_absolute_tolerance():
    audit = audit_special_limits(lmax=4, tolerance=1e-10, absolute_tolerance=1e-12)
    all_equal_appp = next(
        result
        for result in audit.results
        if result.case_name == "all_equal" and result.observables == ("A", "P", "P", "P")
    )
    assert all_equal_appp.passed
    assert abs(all_equal_appp.direct) < 1e-18
    assert all_equal_appp.absolute_error < 1e-12
