import sympy as sp

from celestial_intertwiners.fourpoint.kuntz_benchmark import load_kuntz_fixtures
from celestial_intertwiners.fourpoint.kuntz_formula import (
    KUNTZ_VARIABLES,
    evaluate_kuntz_pppp_closed_form,
    evaluate_kuntz_pppp_closed_form_angles,
    kuntz_pppp_ascii_expression,
    kuntz_pppp_sympy_expression,
    verify_kuntz_formula_fixtures,
)


def test_kuntz_master_expression_imports_as_five_variable_formula():
    ascii_expression = kuntz_pppp_ascii_expression()
    assert "Subscript[" not in ascii_expression
    assert all(variable in ascii_expression for variable in KUNTZ_VARIABLES)

    sympy_expression = kuntz_pppp_sympy_expression()
    free_symbols = {str(symbol) for symbol in sympy_expression.free_symbols}
    assert free_symbols == set(KUNTZ_VARIABLES)
    assert isinstance(sympy_expression, sp.Expr)


def test_executable_kuntz_formula_matches_notebook_fixtures():
    ok, max_relative_error = verify_kuntz_formula_fixtures(tolerance=1e-11)
    assert ok
    assert max_relative_error < 1e-11


def test_kuntz_formula_vector_wrapper_matches_angle_evaluator():
    fixture = load_kuntz_fixtures()[0]
    angle_value = evaluate_kuntz_pppp_closed_form_angles(fixture.angles)
    vector_value = evaluate_kuntz_pppp_closed_form(fixture.vectors)

    assert abs(angle_value - fixture.value) / abs(fixture.value) < 1e-11
    assert abs(vector_value - fixture.value) / abs(fixture.value) < 1e-11
