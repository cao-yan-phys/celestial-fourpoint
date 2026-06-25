"""Executable Kuntz four-point closed-form expression.

The external companion notebook ships the full PTA-only ``fourPoint`` formula
as Wolfram Language.  This module imports that expression into SymPy and
provides a numerical evaluator in the same five-angle convention used by the
notebook fixtures.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import sympy as sp
from sympy.parsing.mathematica import parse_mathematica

from .kuntz_benchmark import DEFAULT_FIXTURE_PATH, load_kuntz_fixtures, vec2angles

KUNTZ_VARIABLES = ("cb", "cc", "cd", "psic", "psid")

DEFAULT_KUNTZ_INPUT_PATH = DEFAULT_FIXTURE_PATH.with_name("4PT_inputs.wl")

_SYMBOLS = sp.symbols("cb cc cd psic psid")
_WOLFRAM_REPLACEMENTS = {
    "Subscript[c, b]": "cb",
    "Subscript[c, c]": "cc",
    "Subscript[c, d]": "cd",
    "Subscript[\u03a8, c]": "psic",
    "Subscript[\u03a8, d]": "psid",
}


def _path_key(path: str | Path | None) -> str:
    source = DEFAULT_KUNTZ_INPUT_PATH if path is None else Path(path)
    return str(source.resolve())


def _extract_wolfram_assignment(source: str, name: str) -> str:
    marker = f"{name} ="
    start = source.index(marker) + len(marker)
    end = source.index("; ]", start)
    return source[start:end].strip()


def normalize_kuntz_wolfram_expression(expression: str) -> str:
    """Return a SymPy/ASCII-friendly version of a Kuntz Wolfram expression."""

    normalized = expression
    for wolfram, ascii_name in _WOLFRAM_REPLACEMENTS.items():
        normalized = normalized.replace(wolfram, ascii_name)
    if "Subscript[" in normalized:
        raise ValueError("unrecognized Wolfram subscript in Kuntz expression")
    return normalized


@lru_cache(maxsize=4)
def _cached_ascii_expression(path_key: str) -> str:
    source = Path(path_key).read_text(encoding="utf-8")
    wolfram = _extract_wolfram_assignment(source, "fourPoint")
    return normalize_kuntz_wolfram_expression(wolfram)


def kuntz_pppp_ascii_expression(path: str | Path | None = None) -> str:
    """Return the full Kuntz PPPP master formula in ASCII Mathematica syntax."""

    return _cached_ascii_expression(_path_key(path))


@lru_cache(maxsize=4)
def _cached_sympy_expression(path_key: str) -> sp.Expr:
    return parse_mathematica(_cached_ascii_expression(path_key))


def kuntz_pppp_sympy_expression(path: str | Path | None = None) -> sp.Expr:
    """Return the full Kuntz PPPP master formula as a SymPy expression."""

    return _cached_sympy_expression(_path_key(path))


@lru_cache(maxsize=4)
def _cached_lambdified(path_key: str):
    expression = _cached_sympy_expression(path_key)
    return sp.lambdify(_SYMBOLS, expression, modules="numpy")


def evaluate_kuntz_pppp_closed_form_angles(
    angles: Sequence[float],
    *,
    path: str | Path | None = None,
) -> complex:
    """Evaluate the Kuntz PPPP master at ``(cb, cc, cd, psi_c, psi_d)``."""

    if len(angles) != 5:
        raise ValueError("Kuntz closed form requires five angles")
    evaluator = _cached_lambdified(_path_key(path))
    return complex(evaluator(*angles))


def evaluate_kuntz_pppp_closed_form(
    vectors,
    *,
    path: str | Path | None = None,
) -> complex:
    """Evaluate the Kuntz PPPP master for four unit directions."""

    return evaluate_kuntz_pppp_closed_form_angles(vec2angles(*vectors), path=path)


def verify_kuntz_formula_fixtures(
    *,
    path: str | Path | None = None,
    tolerance: float = 1e-11,
) -> tuple[bool, float]:
    """Check the executable Kuntz formula against all exported fixtures."""

    max_relative_error = 0.0
    for fixture in load_kuntz_fixtures():
        value = evaluate_kuntz_pppp_closed_form_angles(fixture.angles, path=path)
        relative_error = abs(value - fixture.value) / max(abs(fixture.value), 1e-30)
        max_relative_error = max(max_relative_error, float(relative_error))
    return max_relative_error < tolerance, max_relative_error
