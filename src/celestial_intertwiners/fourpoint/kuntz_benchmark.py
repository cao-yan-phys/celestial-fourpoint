"""Kuntz-Smarra-Vaglio companion-notebook benchmark helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from ..geometry import spherical_angles_from_direction
from .conventions import validate_epsilon
from .direct_quadrature import fibonacci_sphere
from .pointwise_descendant import helicity_polarization_from_axis

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "external"
    / "PTA-4PT-correlator"
    / "kuntz_fixtures.json"
)
KUNTZ_SEED_EPSILONS = (-1, 1, -1, 1)


@dataclass(frozen=True)
class KuntzFixture:
    """One deterministic Kuntz notebook evaluation."""

    name: str
    vectors: np.ndarray
    angles: tuple[float, float, float, float, float]
    value: complex


@dataclass(frozen=True)
class KuntzBenchmarkResult:
    """Direct quadrature vs. Kuntz closed-form comparison result."""

    fixture_name: str
    reference: complex
    direct: complex
    relative_error: float
    passed: bool


@dataclass(frozen=True)
class KuntzSweepResult:
    """Direct quadrature comparison for all exported Kuntz fixtures."""

    results: tuple[KuntzBenchmarkResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def max_relative_error(self) -> float:
        if not self.results:
            return math.inf
        return max(result.relative_error for result in self.results)


def _repo_fixture_path(path: str | Path | None = None) -> Path:
    return DEFAULT_FIXTURE_PATH if path is None else Path(path)


def _parse_float(value) -> float:
    if isinstance(value, int | float):
        return float(value)
    return float(str(value).replace("*^", "e").replace("E", "e"))


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if norm == 0:
        raise ValueError("direction vector must be nonzero")
    return array / norm


def _acos_clamped(value: float) -> float:
    return math.acos(max(-1.0, min(1.0, value)))


def vec2angles(p1, p2, p3, p4) -> tuple[float, float, float, float, float]:
    """Return the five Kuntz companion-notebook angles for four directions."""

    p1 = _unit(p1)
    p2 = _unit(p2)
    p3 = _unit(p3)
    p4 = _unit(p4)
    c12 = float(np.dot(p1, p2))
    c13 = float(np.dot(p1, p3))
    c14 = float(np.dot(p1, p4))
    normal = np.cross(p1, p2)

    def psi_for(point, cosine_to_p1: float) -> float:
        denominator = math.sqrt(1.0 - c12 * c12) * math.sqrt(
            1.0 - cosine_to_p1 * cosine_to_p1
        )
        if denominator == 0:
            raise ValueError("Kuntz vec2Angles is singular for this configuration")
        cosine = (float(np.dot(p2, point)) - c12 * cosine_to_p1) / denominator
        sign = float(np.sign(np.dot(point, normal)))
        return sign * _acos_clamped(cosine)

    return (c12, c13, c14, psi_for(p3, c13), psi_for(p4, c14))


@lru_cache(maxsize=8)
def load_kuntz_fixtures(path: str | Path | None = None) -> tuple[KuntzFixture, ...]:
    """Load deterministic values exported from the Kuntz companion notebook."""

    fixture_path = _repo_fixture_path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixtures: list[KuntzFixture] = []
    for record in payload["records"]:
        vectors = np.asarray(
            [[_parse_float(component) for component in row] for row in record["vectors"]],
            dtype=float,
        )
        fixtures.append(
            KuntzFixture(
                name=record["name"],
                vectors=np.asarray([_unit(vector) for vector in vectors]),
                angles=tuple(_parse_float(value) for value in record["angles"]),
                value=complex(_parse_float(record["value_re"]), _parse_float(record["value_im"])),
            )
        )
    return tuple(fixtures)


def kuntz_closed_formula_available(path: str | Path | None = None) -> bool:
    """Return True when companion-notebook fixtures are available locally."""

    return _repo_fixture_path(path).exists()


def evaluate_kuntz_pppp_seed(
    fixture_index: int = 0,
    *,
    path: str | Path | None = None,
) -> complex:
    """Return a Kuntz notebook value for the PPPP helicity seed."""

    return load_kuntz_fixtures(path)[fixture_index].value


def _pppp_product_for_source(
    vectors: np.ndarray,
    omega: np.ndarray,
    epsilons: tuple[int, int, int, int],
) -> complex:
    polarizations = {
        epsilon: helicity_polarization_from_axis(-omega, epsilon)
        for epsilon in set(epsilons)
    }
    value = 1.0 + 0.0j
    for vector, epsilon in zip(vectors, epsilons):
        denominator = 1.0 - float(np.dot(omega, vector))
        if abs(denominator) < 1e-13:
            return 0.0j
        numerator = np.einsum("i,j,ij->", vector, vector, polarizations[epsilon])
        value *= 0.5 * numerator / denominator
    return value


def pppp_direct_fibonacci_average(
    vectors,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    n_points: int = 32768,
) -> complex:
    """Directly average the PPPP seed over source directions with Fibonacci points."""

    if len(epsilons) != 4:
        raise ValueError("PPPP benchmark requires four helicities")
    epsilons = tuple(validate_epsilon(epsilon) for epsilon in epsilons)
    unit_vectors = np.asarray([_unit(vector) for vector in vectors], dtype=float)
    total = 0.0j
    for omega in fibonacci_sphere(n_points):
        total += _pppp_product_for_source(unit_vectors, omega, epsilons)
    return total / n_points


def pppp_direct_adaptive_average(
    vectors,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    inner_tolerance: float = 1e-4,
    outer_tolerance: float = 1e-4,
    limit: int = 120,
) -> complex:
    """Directly average the PPPP seed with singularity-aware adaptive quadrature."""

    if len(epsilons) != 4:
        raise ValueError("PPPP benchmark requires four helicities")
    epsilons = tuple(validate_epsilon(epsilon) for epsilon in epsilons)
    unit_vectors = np.asarray([_unit(vector) for vector in vectors], dtype=float)
    angles = [spherical_angles_from_direction(vector) for vector in unit_vectors]
    phi_breaks = sorted(
        phi
        for _theta, phi in ((theta, float(phi % math.tau)) for theta, phi in angles)
        if 0.0 < phi < math.tau
    )
    x_breaks = sorted(
        float(vector[2]) for vector in unit_vectors if -1.0 < float(vector[2]) < 1.0
    )

    def value_at(x: float, phi: float) -> complex:
        radius = math.sqrt(max(0.0, 1.0 - x * x))
        omega = np.array([radius * math.cos(phi), radius * math.sin(phi), x])
        return _pppp_product_for_source(unit_vectors, omega, epsilons)

    def integrate_part(imaginary: bool) -> float:
        def inner_integral(x: float) -> float:
            def phi_integrand(phi: float) -> float:
                value = value_at(x, phi)
                return float(value.imag if imaginary else value.real)

            integral, _error = quad(
                phi_integrand,
                0.0,
                math.tau,
                points=phi_breaks,
                epsabs=inner_tolerance,
                epsrel=inner_tolerance,
                limit=limit,
            )
            return float(integral)

        integral, _error = quad(
            inner_integral,
            -1.0,
            1.0,
            points=x_breaks,
            epsabs=outer_tolerance,
            epsrel=outer_tolerance,
            limit=limit,
        )
        return float(integral)

    return complex(integrate_part(False), integrate_part(True)) / (4.0 * math.pi)


@lru_cache(maxsize=8)
def benchmark_kuntz_fixture(
    fixture_index: int = 0,
    *,
    path: str | Path | None = None,
    tolerance: float = 1e-8,
    inner_tolerance: float = 1e-4,
    outer_tolerance: float = 1e-4,
) -> KuntzBenchmarkResult:
    """Compare adaptive direct PPPP quadrature against one Kuntz fixture."""

    fixture = load_kuntz_fixtures(path)[fixture_index]
    direct = pppp_direct_adaptive_average(
        fixture.vectors,
        KUNTZ_SEED_EPSILONS,
        inner_tolerance=inner_tolerance,
        outer_tolerance=outer_tolerance,
    )
    scale = max(abs(fixture.value), 1e-30)
    relative_error = abs(direct - fixture.value) / scale
    return KuntzBenchmarkResult(
        fixture_name=fixture.name,
        reference=fixture.value,
        direct=direct,
        relative_error=float(relative_error),
        passed=relative_error < tolerance,
    )


def benchmark_all_kuntz_fixtures(
    *,
    tolerance: float = 5e-6,
    inner_tolerance: float = 1e-4,
    outer_tolerance: float = 1e-4,
) -> KuntzSweepResult:
    """Run the slower adaptive direct-vs-notebook sweep over all fixtures."""

    results = tuple(
        benchmark_kuntz_fixture(
            index,
            tolerance=tolerance,
            inner_tolerance=inner_tolerance,
            outer_tolerance=outer_tolerance,
        )
        for index, _fixture in enumerate(load_kuntz_fixtures())
    )
    return KuntzSweepResult(results=results)


def verify_kuntz_companion_permutation_fixtures(*, tolerance: float = 1e-12) -> bool:
    """Check the exact permutation pattern encoded in the companion fixtures."""

    fixtures = load_kuntz_fixtures()
    if len(fixtures) < 4:
        return False
    values = tuple(fixture.value for fixture in fixtures[:4])
    scale = max(abs(values[0]), 1e-30)
    return (
        abs(values[1] - values[0]) / scale < tolerance
        and abs(values[2] - values[0]) / scale < tolerance
        and abs(values[3] - values[0].conjugate()) / scale < tolerance
    )


@lru_cache(maxsize=8)
def verify_pppp_direct_vs_kuntz_fixture(
    fixture_index: int = 0,
    *,
    tolerance: float = 1e-8,
    inner_tolerance: float = 1e-4,
    outer_tolerance: float = 1e-4,
) -> bool:
    """Return True when the adaptive direct seed matches the notebook fixture."""

    if not kuntz_closed_formula_available():
        return False
    kwargs = {"tolerance": tolerance}
    if inner_tolerance != 1e-4:
        kwargs["inner_tolerance"] = inner_tolerance
    if outer_tolerance != 1e-4:
        kwargs["outer_tolerance"] = outer_tolerance
    return benchmark_kuntz_fixture(
        fixture_index,
        **kwargs,
    ).passed
