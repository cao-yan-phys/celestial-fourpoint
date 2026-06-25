"""Fast physical four-point evaluator with optional CUDA acceleration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from ..finite_kernel_check import canonical_helicity_dyad
from ..geometry import spherical_angles_from_direction
from .conventions import normalize_observable, validate_epsilon
from .kuntz_benchmark import KUNTZ_SEED_EPSILONS

ASTROMETRY_DESCENDANT_NORMALIZATION = -2.0 * math.sqrt(2.0)


@dataclass(frozen=True)
class FastEvaluationResult:
    """One fast physical four-point evaluation."""

    observables: tuple[str, str, str, str]
    value: complex
    n_theta: int
    n_phi: int
    backend: str


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("direction must be nonzero")
    return vector / norm


def _validate_observables(observables) -> tuple[str, str, str, str]:
    if len(observables) != 4:
        raise ValueError("four observables are required")
    normalized = tuple(normalize_observable(observable) for observable in observables)
    if any(observable not in {"P", "A"} for observable in normalized):
        raise ValueError("fast evaluator supports only P and A observables")
    return normalized


def _validate_epsilons(epsilons) -> tuple[int, int, int, int]:
    if len(epsilons) != 4:
        raise ValueError("four helicities are required")
    return tuple(validate_epsilon(epsilon) for epsilon in epsilons)


def _array_module(backend: str):
    key = backend.lower()
    if key in {"cpu", "numpy"}:
        return np, "numpy"
    if key in {"cuda", "gpu", "cupy"}:
        import cupy as cp

        return cp, "cupy"
    if key == "auto":
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() > 0:
                return cp, "cupy"
        except Exception:
            pass
        return np, "numpy"
    raise ValueError("backend must be 'numpy', 'cupy', 'cuda', 'gpu', or 'auto'")


def _to_complex(value, xp) -> complex:
    if xp is np:
        return complex(value)
    return complex(value.get())


class FastFourPointEvaluator:
    """Vectorized physical P/A helicity four-point evaluator.

    The CPU path uses NumPy. The CUDA path is activated with ``backend="cupy"``
    or ``backend="cuda"`` and requires CuPy to be installed by the caller.
    """

    def __init__(self, *, n_theta: int = 80, n_phi: int = 160, backend: str = "auto"):
        if n_theta <= 0 or n_phi <= 0:
            raise ValueError("n_theta and n_phi must be positive")
        self.n_theta = int(n_theta)
        self.n_phi = int(n_phi)
        self.xp, self.backend = _array_module(backend)
        self.omega, self.weights = self._build_source_grid()

    def _build_source_grid(self):
        xs, theta_weights = leggauss(self.n_theta)
        phis = np.linspace(0.0, 2.0 * math.pi, self.n_phi, endpoint=False)
        x_grid, phi_grid = np.meshgrid(xs, phis, indexing="ij")
        radius = np.sqrt(np.maximum(0.0, 1.0 - x_grid * x_grid))
        omega = np.stack(
            [
                radius * np.cos(phi_grid),
                radius * np.sin(phi_grid),
                x_grid,
            ],
            axis=-1,
        ).reshape(-1, 3)
        weights = np.repeat(theta_weights, self.n_phi) * (2.0 * math.pi / self.n_phi)
        return self.xp.asarray(omega), self.xp.asarray(weights)

    def _polarization_axes(self, omega):
        """Return the no-twist x/y polarization axes at axis direction -omega."""

        xp = self.xp
        q = -omega
        qx = q[:, 0]
        qy = q[:, 1]
        qz = q[:, 2]
        axis_x = -qy
        axis_y = qx
        sin2 = axis_x * axis_x + axis_y * axis_y
        factor = xp.where(sin2 > 1e-28, (1.0 - qz) / sin2, 0.0)

        x_axis = xp.stack(
            [
                1.0 - factor * axis_y * axis_y,
                factor * axis_x * axis_y,
                -axis_y,
            ],
            axis=1,
        )
        y_axis = xp.stack(
            [
                factor * axis_x * axis_y,
                1.0 - factor * axis_x * axis_x,
                axis_x,
            ],
            axis=1,
        )
        south_pole = sin2 <= 1e-28
        if bool(xp.any(south_pole)):
            negative_z = south_pole & (qz < 0)
            y_axis = y_axis.copy()
            y_axis[negative_z, 1] = -1.0
        return x_axis, y_axis

    def _pta_antenna(self, p_vector, epsilon: int):
        xp = self.xp
        p = xp.asarray(_unit(p_vector))
        x_axis, y_axis = self._polarization_axes(self.omega)
        p_dot_x = x_axis @ p
        p_dot_y = y_axis @ p
        numerator = p_dot_x * p_dot_x - p_dot_y * p_dot_y + 2j * epsilon * p_dot_x * p_dot_y
        denominator = 1.0 - (self.omega @ p)
        return 0.5 * numerator / denominator

    def _astrometry_antenna(self, p_vector, epsilon: int):
        xp = self.xp
        p_np = _unit(p_vector)
        p = xp.asarray(p_np)
        x_axis, y_axis = self._polarization_axes(self.omega)
        p_dot_x = x_axis @ p
        p_dot_y = y_axis @ p
        nn_e = p_dot_x * p_dot_x - p_dot_y * p_dot_y + 2j * epsilon * p_dot_x * p_dot_y
        e_n = (
            x_axis * p_dot_x[:, None]
            - y_axis * p_dot_y[:, None]
            + 1j * epsilon * (x_axis * p_dot_y[:, None] + y_axis * p_dot_x[:, None])
        )
        denominator = 1.0 - (self.omega @ p)
        response = 0.5 * (((p - self.omega) / denominator[:, None]) * nn_e[:, None] - e_n)
        component_dyad = xp.asarray(canonical_helicity_dyad(p_np, sign=-epsilon))
        component = response @ component_dyad
        normalized = component / ASTROMETRY_DESCENDANT_NORMALIZATION
        # Match the spectral inverse-descendant convention used for mixed kernels.
        return epsilon * normalized

    def antenna(self, observable: str, p_vector, epsilon: int):
        """Return antenna values over the evaluator's source grid."""

        observable = normalize_observable(observable)
        epsilon = validate_epsilon(epsilon)
        if observable == "P":
            return self._pta_antenna(p_vector, epsilon)
        if observable == "A":
            return self._astrometry_antenna(p_vector, epsilon)
        raise ValueError("observable must be P or A")

    def evaluate(
        self,
        observables,
        vectors,
        epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    ) -> FastEvaluationResult:
        """Evaluate one physical P/A four-point kernel by vectorized quadrature."""

        observables = _validate_observables(observables)
        epsilons = _validate_epsilons(epsilons)
        if len(vectors) != 4:
            raise ValueError("four directions are required")
        product = self.xp.ones(self.weights.shape, dtype=complex)
        for observable, vector, epsilon in zip(observables, vectors, epsilons):
            product *= self.antenna(observable, vector, epsilon)
        value = self.xp.sum(self.weights * product) / (4.0 * math.pi)
        return FastEvaluationResult(
            observables=observables,
            value=_to_complex(value, self.xp),
            n_theta=self.n_theta,
            n_phi=self.n_phi,
            backend=self.backend,
        )

    def evaluate_family(
        self,
        vectors,
        epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    ) -> dict[tuple[str, str, str, str], complex]:
        """Evaluate the PPPP-to-AAAA descendant family for fixed directions."""

        cases = (
            ("P", "P", "P", "P"),
            ("A", "P", "P", "P"),
            ("A", "A", "P", "P"),
            ("A", "A", "A", "P"),
            ("A", "A", "A", "A"),
        )
        return {case: self.evaluate(case, vectors, epsilons).value for case in cases}


def fast_physical_mixed_average(
    observables,
    vectors,
    epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    *,
    n_theta: int = 80,
    n_phi: int = 160,
    backend: str = "auto",
) -> complex:
    """Convenience wrapper for one fast physical mixed-kernel evaluation."""

    evaluator = FastFourPointEvaluator(n_theta=n_theta, n_phi=n_phi, backend=backend)
    return evaluator.evaluate(observables, vectors, epsilons).value
