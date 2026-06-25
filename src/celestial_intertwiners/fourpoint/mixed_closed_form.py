"""Symbolic scaffolding for mixed closed-form generation.

The Kuntz master is a scalar function of
``(cb, cc, cd, psic, psid)``.  For the ``c`` and ``d`` legs these variables are
ordinary polar coordinates around the ``a``-leg axis, with ``b`` fixing the
azimuthal origin.  This module gives the local spin descendant operator and its
Fourier-mode inverse as symbolic building blocks for generating mixed kernels
from the executable Kuntz master.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss

from .conventions import validate_epsilon
from .direct_quadrature import sph_harm_y, spin_weighted_spherical_harmonic
from .kuntz_benchmark import KUNTZ_SEED_EPSILONS, load_kuntz_fixtures
from .kuntz_formula import kuntz_pppp_sympy_expression
from .kuntz_formula import evaluate_kuntz_pppp_closed_form_angles
from .kuntz_formula import _cached_lambdified, _path_key

CB, CC, CD, PSIC, PSID = sp.symbols("cb cc cd psic psid")
KUNTZ_SYMBOLS = (CB, CC, CD, PSIC, PSID)
ZC, ZD, QC, UC = sp.symbols("zc zd qc uc")
YC = sp.symbols("yc")


@dataclass(frozen=True)
class LegCoordinates:
    """Coordinate symbols for one Kuntz external leg."""

    leg: str
    cosine: sp.Symbol
    azimuth: sp.Symbol


@dataclass(frozen=True)
class InverseDescendantModeFormula:
    """Fourier-mode Green formula for one inverse descendant."""

    mode: int
    epsilon: int
    cosine: sp.Symbol
    dummy: sp.Symbol
    integrating_factor: sp.Expr
    source_integrand: sp.Expr
    solution: sp.Expr


@dataclass(frozen=True)
class ResolvedLogCoefficient:
    """One explicitly integrated mixed closed-form log coefficient."""

    log_argument: str
    source_characteristic: sp.Expr
    coefficient_characteristic: sp.Expr
    coefficient_coordinate: sp.Expr


@dataclass(frozen=True)
class ResolvedULogPrimitive:
    """One c-leg log coefficient primitive in half-angle characteristic form."""

    log_argument: str
    source_u: sp.Expr
    primitive_u: sp.Expr
    coefficient_u: sp.Expr


@dataclass(frozen=True)
class ResolvedURationalPrimitive:
    """One c-leg rational residue primitive in half-angle characteristic form."""

    term_index: int
    source_u: sp.Expr
    primitive_u: sp.Expr
    coefficient_u: sp.Expr


@dataclass(frozen=True)
class ResolvedUCorrectionPrimitive:
    """One integration-by-parts correction primitive for a c-dependent log."""

    log_argument: str
    source_u: sp.Expr
    source_y: sp.Expr
    primitive_y: sp.Expr
    primitive_u: sp.Expr


@dataclass(frozen=True)
class ResolvedUFullPrimitive:
    """Complete c-leg APPP primitive in half-angle characteristic form."""

    primitive_u: sp.Expr
    response_u: sp.Expr


@dataclass(frozen=True)
class APPPCClosedFormNumericalAudit:
    """Numerical audit for the APPP(c) closed-form representative."""

    representative: complex
    direct: complex
    descendant: complex
    pppp_reference: complex
    direct_relative_error: float
    descendant_relative_error: float


@dataclass(frozen=True)
class APPPCProjectedNumericalAudit:
    """Numerical audit for the projected physical APPP(c) kernel."""

    representative: complex
    projected: complex
    direct_canonical: complex
    pppp_reconstruction: complex
    pppp_reference: complex
    representative_relative_error: float
    projected_relative_error: float
    pppp_reconstruction_relative_error: float
    lmax: int
    n_theta: int
    n_phi: int


def leg_coordinates(leg: str) -> LegCoordinates:
    """Return the Kuntz coordinate symbols for leg ``c`` or ``d``."""

    normalized = leg.lower()
    if normalized == "c":
        return LegCoordinates(leg="c", cosine=CC, azimuth=PSIC)
    if normalized == "d":
        return LegCoordinates(leg="d", cosine=CD, azimuth=PSID)
    raise ValueError("only Kuntz coordinate legs 'c' and 'd' are supported")


def coordinate_descendant_operator(expression: sp.Expr, leg: str, epsilon: int) -> sp.Expr:
    """Apply the local ``D^epsilon`` operator to one astrometric leg.

    The input expression is interpreted as a spin ``-epsilon`` component on
    the selected leg.  With ``x = cos(theta)`` and ``phi`` the Kuntz azimuth,

    ``D^epsilon f = sqrt(1-x^2) d_x f
                    - i epsilon / sqrt(1-x^2) d_phi f
                    - x / sqrt(1-x^2) f``.
    """

    epsilon = validate_epsilon(epsilon)
    coordinates = leg_coordinates(leg)
    x = coordinates.cosine
    phi = coordinates.azimuth
    sine = sp.sqrt(1 - x**2)
    return sp.simplify(
        sine * sp.diff(expression, x)
        - sp.I * epsilon * sp.diff(expression, phi) / sine
        - x * expression / sine
    )


def inverse_descendant_mode_residual(
    candidate_mode: sp.Expr,
    source_mode: sp.Expr,
    *,
    mode: int,
    epsilon: int,
    cosine: sp.Symbol,
) -> sp.Expr:
    """Return the scalar ODE residual for one inverse-descendant Fourier mode."""

    epsilon = validate_epsilon(epsilon)
    x = cosine
    return sp.simplify(
        (1 - x**2) * sp.diff(candidate_mode, x)
        + (epsilon * mode - x) * candidate_mode
        - sp.sqrt(1 - x**2) * source_mode
    )


def inverse_descendant_integrating_factor(
    *,
    mode: int,
    epsilon: int,
    cosine: sp.Symbol,
) -> sp.Expr:
    """Return the Fourier-mode integrating factor for ``D^{-1}``."""

    epsilon = validate_epsilon(epsilon)
    x = cosine
    return sp.sqrt(1 - x**2) * ((1 + x) / (1 - x)) ** (sp.Rational(epsilon * mode, 2))


def inverse_descendant_mode_formula(
    source_mode: sp.Expr,
    *,
    mode: int,
    epsilon: int,
    cosine: sp.Symbol,
    integration_constant: sp.Expr | None = None,
) -> InverseDescendantModeFormula:
    """Return the exact Green/integral solution for one Fourier source mode.

    The constant is intentionally explicit.  In the physical mixed kernel it is
    fixed by the ``ell >= 2`` projection and regularity/special-limit audits.
    """

    epsilon = validate_epsilon(epsilon)
    x = cosine
    t = sp.Dummy(f"{x.name}_int")
    constant = sp.Symbol(f"C_{mode}") if integration_constant is None else integration_constant
    factor = inverse_descendant_integrating_factor(mode=mode, epsilon=epsilon, cosine=x)
    source_at_t = source_mode.xreplace({x: t})
    weight = ((1 + t) / (1 - t)) ** (sp.Rational(epsilon * mode, 2))
    integrand = sp.simplify(weight * source_at_t)
    solution = (constant + sp.Integral(integrand, t)) / factor
    return InverseDescendantModeFormula(
        mode=mode,
        epsilon=epsilon,
        cosine=x,
        dummy=t,
        integrating_factor=factor,
        source_integrand=integrand,
        solution=solution,
    )


def kuntz_master_descendant_source() -> sp.Expr:
    """Return the executable Kuntz master as the source for mixed generation."""

    return kuntz_pppp_sympy_expression()


def _c_leg_denominator() -> sp.Expr:
    return sp.sqrt(1 - CD**2) * (1 - CB) + ZD * sp.sqrt(1 - CB**2) * (CD - 1)


def _c_leg_characteristic_to_coordinate(expression: sp.Expr, epsilon: int) -> sp.Expr:
    epsilon = validate_epsilon(epsilon)
    ratio = (1 + CC) / (1 - CC)
    z_value = sp.exp(sp.I * PSIC)
    zd_value = sp.exp(sp.I * PSID)
    q_value = z_value * ratio ** sp.Rational(-epsilon, 2)
    return expression.subs({QC: q_value, ZD: zd_value})


def appp_c_leg_independent_log_coefficients() -> tuple[ResolvedLogCoefficient, ...]:
    """Return the explicitly integrated `c`-leg APPP coefficients for spectator logs.

    These are the two Kuntz logarithms independent of the `c`-leg coordinates,
    `log(2/(1-cb))` and `log(2/(1-cd))`.  For the Kuntz seed helicities
    `(-,+,-,+)`, the `c` leg has `epsilon=-1`; in the characteristic coordinate
    `qc = exp(i psic) sqrt((1+cc)/(1-cc))`, the inverse descendant reduces to
    a single ordinary integral.
    """

    x = CC
    q = QC
    zd = ZD
    sb = sp.sqrt(1 - CB**2)
    sd = sp.sqrt(1 - CD**2)
    sx = sp.sqrt(1 - x**2)
    denominator = _c_leg_denominator()

    cb_poly = (CB + 1) * q**2 + 2 * sb * q + 1 - CB
    cb_linear = (1 - CB) * (1 + x) + sb * q * (x - 1)
    cb_integral = ((1 - CB) - sb * q) * x + ((1 - CB) + sb * q) * x**2 / 2
    cb_prefactor = (1 - CB) * (CD**2 - 1) * cb_poly / (
        8 * q * zd * (CB + 1) * denominator
    )
    cb_source = sp.factor(cb_prefactor * cb_linear)
    cb_coefficient = sp.factor(cb_prefactor * cb_integral / sx)

    cd_poly = (CD + 1) * q**2 + 2 * sd * q * zd - (CD - 1) * zd**2
    cd_linear = -(CD - 1) * zd * (1 + x) - sd * q * (1 - x)
    cd_integral = (-(CD - 1) * zd - sd * q) * x + (
        -(CD - 1) * zd + sd * q
    ) * x**2 / 2
    cd_prefactor = (CB**2 - 1) * (CD - 1) * cd_poly / (
        8 * q * (CD + 1) * denominator
    )
    cd_source = sp.factor(cd_prefactor * cd_linear)
    cd_coefficient = sp.factor(cd_prefactor * cd_integral / sx)

    return (
        ResolvedLogCoefficient(
            log_argument="2/(1 - cb)",
            source_characteristic=cb_source,
            coefficient_characteristic=cb_coefficient,
            coefficient_coordinate=_c_leg_characteristic_to_coordinate(cb_coefficient, -1),
        ),
        ResolvedLogCoefficient(
            log_argument="2/(1 - cd)",
            source_characteristic=cd_source,
            coefficient_characteristic=cd_coefficient,
            coefficient_coordinate=_c_leg_characteristic_to_coordinate(cd_coefficient, -1),
        ),
    )


@lru_cache(maxsize=1)
def _kuntz_log_coefficients() -> dict[sp.Expr, sp.Expr]:
    master = kuntz_pppp_sympy_expression()
    logs = sorted(master.atoms(sp.log), key=str)
    collected = sp.collect(master, logs, evaluate=False)
    return {log: collected[log] for log in logs}


@lru_cache(maxsize=1)
def _kuntz_rational_terms() -> tuple[sp.Expr, ...]:
    master = kuntz_pppp_sympy_expression()
    logs = sorted(master.atoms(sp.log), key=str)
    collected = sp.collect(master, logs, evaluate=False)
    return sp.Add.make_args(collected[sp.Integer(1)])


def _physical_c_leg_branch_substitutions(sine_symbol: sp.Symbol) -> dict[sp.Expr, sp.Expr]:
    sb = sp.sqrt(1 - CB**2)
    sd = sp.sqrt(1 - CD**2)
    return {
        sp.sqrt(1 - CC**2): sine_symbol,
        sp.sqrt((CB**2 - 1) * (CC**2 - 1)): sb * sine_symbol,
        sp.sqrt((CC**2 - 1) * (CD**2 - 1)): sine_symbol * sd,
        sp.sqrt((CB**2 - 1) * (CD**2 - 1)): sb * sd,
        sp.sqrt((CB**2 - 1) ** 2 * (CC**2 - 1) * (CD**2 - 1)): (
            (1 - CB**2) * sine_symbol * sd
        ),
        sp.sqrt((CB**2 - 1) ** 3 * (CC**2 - 1) * (CD**2 - 1) ** 2): (
            sb**3 * sine_symbol * (1 - CD**2)
        ),
        sp.sqrt((CB**2 - 1) ** 3 * (CC**2 - 1)): sb**3 * sine_symbol,
        sp.sqrt((CB**2 - 1) * (CC**2 - 1) * (CD**2 - 1) ** 2): (
            sb * sine_symbol * (1 - CD**2)
        ),
        sp.sqrt((CB**2 - 1) ** 2 * (CC**2 - 1) * (CD**2 - 1) ** 3): (
            (1 - CB**2) * sine_symbol * sd**3
        ),
    }


def _exp_to_z(expression: sp.Expr) -> sp.Expr:
    def convert(node):
        if node.func is sp.exp and node.args[0].has(PSIC):
            argument = sp.expand(node.args[0] / sp.I)
            n_c = sp.simplify(argument.coeff(PSIC))
            n_d = sp.simplify(argument.coeff(PSID))
            rest = sp.simplify(argument - n_c * PSIC - n_d * PSID)
            if rest == 0 and n_c.is_integer and n_d.is_integer:
                return ZC ** int(n_c) * ZD ** int(n_d)
        return node

    converted = expression.replace(
        lambda node: node.func is sp.exp and node.args[0].has(PSIC),
        convert,
    )
    return converted.xreplace(
        {
            sp.cos(PSIC): (ZC + 1 / ZC) / 2,
            sp.cos(PSIC - PSID): (ZC / ZD + ZD / ZC) / 2,
        }
    )


def _c_leg_u_source_integrand(source: sp.Expr) -> sp.Expr:
    sine = sp.Symbol("s_c")
    x_of_u = (UC**2 - 1) / (UC**2 + 1)
    branched = source.xreplace(_physical_c_leg_branch_substitutions(sine))
    z_expression = _exp_to_z(branched)
    substitutions = {
        CC: x_of_u,
        sine: 2 * UC / (UC**2 + 1),
        ZC: QC / UC,
        sp.exp(sp.I * PSID): ZD,
        sp.exp(2 * sp.I * PSID): ZD**2,
        sp.exp(3 * sp.I * PSID): ZD**3,
        sp.exp(4 * sp.I * PSID): ZD**4,
    }
    source_u = z_expression.subs(substitutions)
    return sp.cancel(source_u * sp.diff(x_of_u, UC))


def _c_leg_log_argument_u(argument: sp.Expr) -> sp.Expr:
    sine = sp.Symbol("s_c")
    x_of_u = (UC**2 - 1) / (UC**2 + 1)
    branched = argument.xreplace(_physical_c_leg_branch_substitutions(sine))
    z_expression = _exp_to_z(branched)
    return sp.cancel(
        z_expression.subs(
            {
                CC: x_of_u,
                sine: 2 * UC / (UC**2 + 1),
                ZC: QC / UC,
                sp.exp(sp.I * PSID): ZD,
                sp.exp(2 * sp.I * PSID): ZD**2,
                sp.exp(3 * sp.I * PSID): ZD**3,
                sp.exp(4 * sp.I * PSID): ZD**4,
            }
        )
    )


def _x_characteristic_to_u(expression: sp.Expr) -> sp.Expr:
    sine = sp.Symbol("s_c")
    x_of_u = (UC**2 - 1) / (UC**2 + 1)
    return sp.cancel(
        expression.xreplace({sp.sqrt(1 - CC**2): sine}).subs(
            {
                CC: x_of_u,
                sine: 2 * UC / (UC**2 + 1),
            }
        )
    )


@lru_cache(maxsize=16)
def _u_basis_integral(power: int, denominator_power: int) -> sp.Expr:
    return sp.integrate(UC**power / (1 + UC**2) ** denominator_power, UC)


def _integrate_c_leg_u_rational(source_u: sp.Expr) -> sp.Expr:
    numerator, denominator = sp.fraction(source_u)
    common = None
    denominator_power = None
    for power in range(0, 9):
        candidate = sp.cancel(denominator / (1 + UC**2) ** power)
        if not candidate.has(UC):
            common = candidate
            denominator_power = power
            break
    if common is None or denominator_power is None:
        raise ValueError("expected a c-leg denominator proportional to a power of (1 + uc^2)")
    polynomial = sp.Poly(numerator, UC)
    primitive = sum(
        coefficient * _u_basis_integral(powers[0], denominator_power)
        for powers, coefficient in polynomial.terms()
    )
    return primitive / common


def _integrate_correction_source_y(source_y: sp.Expr) -> sp.Expr:
    numerator, denominator = sp.fraction(source_y)

    # The numerator and the non-(yc+1)^3 denominator factor are both linear.
    # Reading the coefficients by sampling avoids the expensive generic
    # polynomial-domain normalization that SymPy attempts for large EX domains.
    n0 = numerator.subs(YC, 0)
    n1 = numerator.subs(YC, 1) - n0
    b = denominator.subs(YC, 0)
    a = denominator.subs(YC, 1) / 8 - b

    pole_gap = b - a
    delta = n1 * b - a * n0
    c3 = (n0 - n1) / pole_gap
    c2 = delta / pole_gap**2
    c1 = -a * delta / pole_gap**3

    linear = a * YC + b
    return (
        c1 * (sp.log(YC + 1) - sp.log(linear))
        - c2 / (YC + 1)
        - c3 / (2 * (YC + 1) ** 2)
    )


def _integrate_correction_linear_y(source_u: sp.Expr) -> sp.Expr:
    """Integrate ``source_u du`` via ``yc = uc^2`` with fixed partial fractions.

    The correction terms have the form

    ``N(yc) / ((yc + 1)^3 (A yc + B)) dyc``

    after the substitution ``yc = uc^2``.  The numerator is at most linear, so
    the four partial-fraction coefficients are fixed by closed formulas rather
    than by a general CAS decomposition.
    """

    return _integrate_correction_source_y(_correction_source_y(source_u))


def _correction_source_y(source_u: sp.Expr) -> sp.Expr:
    source_y = sp.cancel((source_u / UC).subs(UC, sp.sqrt(YC)) / 2)
    if source_y.has(sp.sqrt(YC)):
        raise ValueError("correction source did not become rational in yc")
    return source_y


def appp_c_leg_dependent_log_primitives() -> tuple[ResolvedULogPrimitive, ...]:
    """Return explicit primitives for the two c-dependent APPP log coefficients."""

    results: list[ResolvedULogPrimitive] = []
    for log, coefficient in _kuntz_log_coefficients().items():
        if not (log.has(CC) or log.has(PSIC)):
            continue
        source_u = _c_leg_u_source_integrand(coefficient)
        primitive_u = _integrate_c_leg_u_rational(source_u)
        coefficient_u = primitive_u * (1 + UC**2) / (2 * UC)
        results.append(
            ResolvedULogPrimitive(
                log_argument=str(log.args[0]),
                source_u=source_u,
                primitive_u=primitive_u,
                coefficient_u=coefficient_u,
            )
        )
    return tuple(results)


def verify_appp_c_dependent_log_primitives() -> bool:
    """Check the two c-dependent APPP log coefficient primitives exactly."""

    for primitive in appp_c_leg_dependent_log_primitives():
        if sp.cancel(sp.diff(primitive.primitive_u, UC) - primitive.source_u) != 0:
            return False
        if primitive.primitive_u.has(sp.Integral):
            return False
    return True


@lru_cache(maxsize=1)
def appp_c_leg_log_correction_primitives() -> tuple[ResolvedUCorrectionPrimitive, ...]:
    """Return the two integration-by-parts correction primitives for APPP(c)."""

    logs = [log for log in _kuntz_log_coefficients() if log.has(CC) or log.has(PSIC)]
    primitives = appp_c_leg_dependent_log_primitives()
    results: list[ResolvedUCorrectionPrimitive] = []
    for log, primitive in zip(logs, primitives):
        argument_u = _c_leg_log_argument_u(log.args[0])
        source_u = sp.cancel(primitive.primitive_u * sp.diff(argument_u, UC) / argument_u)
        source_y = _correction_source_y(source_u)
        primitive_y = _integrate_correction_source_y(source_y)
        primitive_u = primitive_y.subs(YC, UC**2)
        results.append(
            ResolvedUCorrectionPrimitive(
                log_argument=str(log.args[0]),
                source_u=source_u,
                source_y=source_y,
                primitive_y=primitive_y,
                primitive_u=primitive_u,
            )
        )
    return tuple(results)


def verify_appp_c_log_correction_primitives() -> bool:
    """Check the two APPP(c) log correction primitives structurally."""

    for primitive in appp_c_leg_log_correction_primitives():
        if primitive.primitive_u.has(sp.Integral):
            return False
        if primitive.source_y.has(sp.sqrt(YC)):
            return False
        numerator, denominator = sp.fraction(primitive.source_y)
        numerator_2 = numerator.subs(YC, 2)
        numerator_linear_at_2 = 2 * numerator.subs(YC, 1) - numerator.subs(YC, 0)
        denominator_2 = denominator.subs(YC, 2)
        denominator_linear_factor_at_2 = denominator_2 / 27
        denominator_linear_at_2 = (
            2 * denominator.subs(YC, 1) / 8 - denominator.subs(YC, 0)
        )
        if numerator_2 != numerator_linear_at_2:
            return False
        if denominator_linear_factor_at_2 != denominator_linear_at_2:
            return False
    return True


def appp_c_leg_rational_residue_primitives() -> tuple[ResolvedURationalPrimitive, ...]:
    """Return explicit primitives for the four non-log Kuntz rational terms."""

    results: list[ResolvedURationalPrimitive] = []
    for index, term in enumerate(_kuntz_rational_terms()):
        source_u = _c_leg_u_source_integrand(term)
        primitive_u = _integrate_c_leg_u_rational(source_u)
        coefficient_u = primitive_u * (1 + UC**2) / (2 * UC)
        results.append(
            ResolvedURationalPrimitive(
                term_index=index,
                source_u=source_u,
                primitive_u=primitive_u,
                coefficient_u=coefficient_u,
            )
        )
    return tuple(results)


def verify_appp_c_rational_residue_primitives() -> bool:
    """Check the four c-leg rational residue primitives exactly."""

    for primitive in appp_c_leg_rational_residue_primitives():
        if sp.cancel(sp.diff(primitive.primitive_u, UC) - primitive.source_u) != 0:
            return False
        if primitive.primitive_u.has(sp.Integral):
            return False
    return True


@lru_cache(maxsize=1)
def appp_c_leg_full_primitive_u() -> ResolvedUFullPrimitive:
    """Assemble the complete APPP(c) inverse-descendant primitive.

    The returned ``primitive_u`` is the numerator primitive ``F`` in
    ``A = F / sqrt(1 - cc^2)``.  In half-angle variables
    ``sqrt(1 - cc^2) = 2 uc / (1 + uc^2)``.
    """

    log_coefficients = _kuntz_log_coefficients()
    dependent_logs = [log for log in log_coefficients if log.has(CC) or log.has(PSIC)]
    spectator_logs = [log for log in log_coefficients if log not in dependent_logs]

    spectator_pieces = []
    for coefficient, log in zip(appp_c_leg_independent_log_coefficients(), spectator_logs):
        primitive_u = _x_characteristic_to_u(
            coefficient.coefficient_characteristic * sp.sqrt(1 - CC**2)
        )
        spectator_pieces.append(primitive_u * sp.log(log.args[0]))

    dependent_pieces = [
        primitive.primitive_u * sp.log(_c_leg_log_argument_u(log.args[0]))
        for primitive, log in zip(appp_c_leg_dependent_log_primitives(), dependent_logs)
    ]
    rational_piece = sum(primitive.primitive_u for primitive in appp_c_leg_rational_residue_primitives())
    correction_piece = sum(primitive.primitive_u for primitive in appp_c_leg_log_correction_primitives())

    primitive_u = sum(spectator_pieces) + sum(dependent_pieces) + rational_piece - correction_piece
    response_u = primitive_u * (1 + UC**2) / (2 * UC)
    return ResolvedUFullPrimitive(primitive_u=primitive_u, response_u=response_u)


@lru_cache(maxsize=1)
def _appp_c_representative_lambdified():
    return sp.lambdify((CB, CD, UC, QC, ZD), appp_c_leg_full_primitive_u().response_u, modules="numpy")


def _appp_c_characteristic_values(angles) -> tuple[float, float, float, complex, complex]:
    cb, cc, cd, psi_c, psi_d = angles
    u = math.sqrt((1.0 + cc) / (1.0 - cc))
    q = cmath.exp(1j * psi_c) * u
    z_d = cmath.exp(1j * psi_d)
    return float(cb), float(cc), float(cd), q, z_d


def kuntz_canonical_vectors_from_angles(angles) -> np.ndarray:
    """Return the Kuntz-frame four directions for five companion angles.

    Scalar PPPP values are rotation invariant, but mixed spin components carry
    an external-frame phase.  Direct APPP(c) checks must therefore use the same
    canonical frame as the Kuntz variables: ``a`` at the north pole and ``b`` on
    the prime meridian.
    """

    cb, cc, cd, psi_c, psi_d = (float(value) for value in angles)
    sb = math.sqrt(max(0.0, 1.0 - cb * cb))
    sc = math.sqrt(max(0.0, 1.0 - cc * cc))
    sd = math.sqrt(max(0.0, 1.0 - cd * cd))
    return np.asarray(
        [
            [0.0, 0.0, 1.0],
            [sb, 0.0, cb],
            [sc * math.cos(psi_c), sc * math.sin(psi_c), cc],
            [sd * math.cos(psi_d), sd * math.sin(psi_d), cd],
        ],
        dtype=float,
    )


def evaluate_appp_c_representative_closed_form_angles(angles) -> complex:
    """Evaluate the assembled APPP(c) right-inverse representative.

    This is the local inverse-descendant representative before the physical
    ``ell >= 2`` homogeneous projection is fixed.
    """

    cb, _cc, cd, q, z_d = _appp_c_characteristic_values(angles)
    u = abs(q)
    return complex(_appp_c_representative_lambdified()(cb, cd, u, q, z_d))


def _projected_appp_c_and_pppp_reconstruction(
    angles,
    *,
    lmax: int,
    n_theta: int | None = None,
    n_phi: int | None = None,
) -> tuple[complex, complex, int, int]:
    if lmax < 1:
        raise ValueError("lmax must be at least 1")
    cb, cc, cd, psi_c, psi_d = (float(value) for value in angles)
    n_theta = 2 * lmax + 20 if n_theta is None else int(n_theta)
    n_phi = 4 * lmax + 40 if n_phi is None else int(n_phi)

    xs, theta_weights = leggauss(n_theta)
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    x_grid, phi_grid = np.meshgrid(xs, phis, indexing="ij")
    theta_grid = np.arccos(x_grid)
    weights = np.repeat(theta_weights, n_phi) * (2.0 * math.pi / n_phi)

    evaluator = _cached_lambdified(_path_key(None))
    source_values = np.asarray(evaluator(cb, x_grid, cd, phi_grid, psi_d), dtype=complex).ravel()
    weighted_source = weights * source_values

    theta_target = math.acos(max(-1.0, min(1.0, cc)))
    pppp_reconstruction = 0.0j
    projected = 0.0j
    for ell in range(0, lmax + 1):
        transfer = math.sqrt(ell * (ell + 1)) if ell >= 1 else None
        for mode in range(-ell, ell + 1):
            scalar_grid = sph_harm_y(ell, mode, theta_grid, phi_grid).ravel()
            coefficient = complex(np.dot(np.conj(scalar_grid), weighted_source))
            pppp_reconstruction += coefficient * spin_weighted_spherical_harmonic(
                ell,
                mode,
                0,
                theta_target,
                psi_c,
            )
            if ell >= 1:
                projected += coefficient / transfer * spin_weighted_spherical_harmonic(
                    ell,
                    mode,
                    1,
                    theta_target,
                    psi_c,
                )
    return projected, pppp_reconstruction, n_theta, n_phi


def evaluate_appp_c_projected_from_kuntz_master_angles(
    angles,
    *,
    lmax: int = 20,
    n_theta: int | None = None,
    n_phi: int | None = None,
) -> complex:
    """Evaluate the physical APPP(c) branch by external spin projection.

    The Kuntz PPPP closed form is already the converged source integral.  This
    routine projects that scalar master on the external ``c``-sphere and
    applies the spectral inverse descendant coefficient ``1/sqrt(l(l+1))``.
    That fixes the homogeneous branch missed by the local primitive.
    """

    projected, _pppp_reconstruction, _n_theta, _n_phi = _projected_appp_c_and_pppp_reconstruction(
        angles,
        lmax=lmax,
        n_theta=n_theta,
        n_phi=n_phi,
    )
    return projected


def appp_c_descendant_closure_residual(
    angles,
    *,
    step: float = 1e-5,
) -> tuple[complex, complex, float]:
    """Finite-difference check that the representative descends to Kuntz PPPP."""

    cb, cc, cd, psi_c, psi_d = (float(value) for value in angles)

    def value_at(x: float, phi: float) -> complex:
        return evaluate_appp_c_representative_closed_form_angles((cb, x, cd, phi, psi_d))

    d_x = (value_at(cc + step, psi_c) - value_at(cc - step, psi_c)) / (2.0 * step)
    d_phi = (value_at(cc, psi_c + step) - value_at(cc, psi_c - step)) / (2.0 * step)
    sine = math.sqrt(1.0 - cc * cc)
    descendant = sine * d_x + 1j * d_phi / sine - cc * value_at(cc, psi_c) / sine
    reference = evaluate_kuntz_pppp_closed_form_angles((cb, cc, cd, psi_c, psi_d))
    relative_error = abs(descendant - reference) / max(abs(reference), 1e-30)
    return descendant, reference, float(relative_error)


def benchmark_appp_c_representative_against_direct(
    *,
    fixture_index: int = 0,
    n_theta: int = 160,
    n_phi: int = 320,
    backend: str = "numpy",
) -> APPPCClosedFormNumericalAudit:
    """Compare APPP(c) representative with direct physical PPAP quadrature."""

    from .fast_evaluator import FastFourPointEvaluator

    fixture = load_kuntz_fixtures()[fixture_index]
    representative = evaluate_appp_c_representative_closed_form_angles(fixture.angles)
    vectors = kuntz_canonical_vectors_from_angles(fixture.angles)
    direct = FastFourPointEvaluator(n_theta=n_theta, n_phi=n_phi, backend=backend).evaluate(
        ("P", "P", "A", "P"),
        vectors,
        KUNTZ_SEED_EPSILONS,
    ).value
    descendant, pppp_reference, descendant_error = appp_c_descendant_closure_residual(fixture.angles)
    direct_error = abs(representative - direct) / max(abs(direct), 1e-30)
    return APPPCClosedFormNumericalAudit(
        representative=representative,
        direct=direct,
        descendant=descendant,
        pppp_reference=pppp_reference,
        direct_relative_error=float(direct_error),
        descendant_relative_error=descendant_error,
    )


def benchmark_appp_c_projected_against_direct(
    *,
    fixture_index: int = 0,
    lmax: int = 20,
    n_theta: int | None = None,
    n_phi: int | None = None,
    direct_n_theta: int = 160,
    direct_n_phi: int = 320,
    backend: str = "numpy",
) -> APPPCProjectedNumericalAudit:
    """Compare the projected physical APPP(c) branch with canonical direct quadrature."""

    from .fast_evaluator import FastFourPointEvaluator

    fixture = load_kuntz_fixtures()[fixture_index]
    projected, pppp_reconstruction, used_theta, used_phi = _projected_appp_c_and_pppp_reconstruction(
        fixture.angles,
        lmax=lmax,
        n_theta=n_theta,
        n_phi=n_phi,
    )
    representative = evaluate_appp_c_representative_closed_form_angles(fixture.angles)
    vectors = kuntz_canonical_vectors_from_angles(fixture.angles)
    direct = FastFourPointEvaluator(
        n_theta=direct_n_theta,
        n_phi=direct_n_phi,
        backend=backend,
    ).evaluate(("P", "P", "A", "P"), vectors, KUNTZ_SEED_EPSILONS).value
    pppp_reference = evaluate_kuntz_pppp_closed_form_angles(fixture.angles)
    return APPPCProjectedNumericalAudit(
        representative=representative,
        projected=projected,
        direct_canonical=direct,
        pppp_reconstruction=pppp_reconstruction,
        pppp_reference=pppp_reference,
        representative_relative_error=float(abs(representative - direct) / max(abs(direct), 1e-30)),
        projected_relative_error=float(abs(projected - direct) / max(abs(direct), 1e-30)),
        pppp_reconstruction_relative_error=float(
            abs(pppp_reconstruction - pppp_reference) / max(abs(pppp_reference), 1e-30)
        ),
        lmax=int(lmax),
        n_theta=used_theta,
        n_phi=used_phi,
    )


def verify_appp_c_full_primitive_components() -> bool:
    """Check all pieces required for the assembled APPP(c) primitive."""

    full = appp_c_leg_full_primitive_u()
    return (
        not full.primitive_u.has(sp.Integral)
        and all(not primitive.primitive_u.has(sp.Integral) for primitive in appp_c_leg_dependent_log_primitives())
        and all(not primitive.primitive_u.has(sp.Integral) for primitive in appp_c_leg_rational_residue_primitives())
        and verify_appp_c_log_correction_primitives()
    )


def verify_appp_c_independent_log_coefficients() -> bool:
    """Check the resolved APPP spectator-log coefficients by exact ODE closure."""

    sx = sp.sqrt(1 - CC**2)
    for coefficient in appp_c_leg_independent_log_coefficients():
        residual = sp.diff(sx * coefficient.coefficient_characteristic, CC)
        if sp.simplify(residual - coefficient.source_characteristic) != 0:
            return False
    return True


def verify_inverse_descendant_green_scaffold() -> bool:
    """Return True when the coordinate inverse-descendant scaffold is consistent."""

    candidate = sp.sqrt(1 - CC**2)
    source = -2 * CC
    return (
        sp.simplify(coordinate_descendant_operator(candidate, "c", 1) - source) == 0
        and sp.simplify(coordinate_descendant_operator(candidate, "c", -1) - source) == 0
        and inverse_descendant_mode_residual(
            candidate,
            source,
            mode=0,
            epsilon=1,
            cosine=CC,
        )
        == 0
    )
