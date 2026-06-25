"""Persistent finite-spectral four-point kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .kuntz_benchmark import KUNTZ_SEED_EPSILONS
from .spectral_convergence import (
    PHYSICAL_PTA_TRANSFER_SCALE,
    _external_coefficients,
    _source_grid,
    _source_harmonic_matrix,
    _unit,
    _validate_epsilons,
    _validate_observables,
)
from .spectral_sum_evaluator import SpectralSumResult


DEFAULT_KERNEL_DIR = Path(__file__).resolve().parents[3] / "output" / "kernels"


@dataclass(frozen=True)
class PrecomputedKernelBuildInfo:
    """Metadata for a persisted finite-spectral kernel."""

    path: Path
    lmax: int
    n_theta: int
    n_phi: int
    mode_count: int
    grid_size: int
    bytes_written: int


class PrecomputedSpectralKernel:
    """Loaded finite-spectral surrogate for fast mixed-kernel evaluation.

    This stores the source-grid quadrature weights and the source-spin harmonic
    matrices for both tensor helicities.  It is a compact finite-spectral
    function, not a gigantic four-leg tensor.
    """

    def __init__(
        self,
        *,
        lmax: int,
        n_theta: int,
        n_phi: int,
        transfer_scale: float,
        modes: tuple[tuple[int, int], ...],
        weights: np.ndarray,
        source_values: dict[int, np.ndarray],
        path: Path | None = None,
    ):
        if lmax < 2:
            raise ValueError("lmax must be at least 2")
        self.lmax = int(lmax)
        self.n_theta = int(n_theta)
        self.n_phi = int(n_phi)
        self.transfer_scale = float(transfer_scale)
        self.modes = tuple((int(ell), int(mode)) for ell, mode in modes)
        self.weights = np.asarray(weights)
        self.source_values = {
            int(spin): np.asarray(values, dtype=complex)
            for spin, values in source_values.items()
        }
        self.path = path
        self._response_cache: dict[tuple[str, int, bytes], np.ndarray] = {}

    @classmethod
    def build(
        cls,
        *,
        lmax: int = 15,
        n_theta: int | None = None,
        n_phi: int | None = None,
        transfer_scale: float = PHYSICAL_PTA_TRANSFER_SCALE,
    ) -> "PrecomputedSpectralKernel":
        """Build a finite-spectral kernel in memory."""

        n_theta = 2 * lmax + 8 if n_theta is None else int(n_theta)
        n_phi = 4 * lmax + 16 if n_phi is None else int(n_phi)
        _theta_grid, _phi_grid, weights = _source_grid(n_theta, n_phi)
        modes_plus, source_plus = _source_harmonic_matrix(lmax, 2, n_theta, n_phi)
        modes_minus, source_minus = _source_harmonic_matrix(lmax, -2, n_theta, n_phi)
        if modes_plus != modes_minus:
            raise ValueError("source harmonic mode ordering differs between helicities")
        return cls(
            lmax=lmax,
            n_theta=n_theta,
            n_phi=n_phi,
            transfer_scale=transfer_scale,
            modes=modes_plus,
            weights=weights,
            source_values={2: source_plus, -2: source_minus},
        )

    def save(self, path: str | Path) -> PrecomputedKernelBuildInfo:
        """Persist this kernel as an uncompressed ``.npz`` bundle."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        ells = np.asarray([ell for ell, _mode in self.modes], dtype=np.int16)
        ms = np.asarray([mode for _ell, mode in self.modes], dtype=np.int16)
        np.savez(
            target,
            lmax=np.asarray(self.lmax, dtype=np.int16),
            n_theta=np.asarray(self.n_theta, dtype=np.int16),
            n_phi=np.asarray(self.n_phi, dtype=np.int16),
            transfer_scale=np.asarray(self.transfer_scale, dtype=float),
            ells=ells,
            ms=ms,
            weights=self.weights,
            source_spin_plus=self.source_values[2],
            source_spin_minus=self.source_values[-2],
        )
        size = target.stat().st_size
        self.path = target
        return PrecomputedKernelBuildInfo(
            path=target,
            lmax=self.lmax,
            n_theta=self.n_theta,
            n_phi=self.n_phi,
            mode_count=len(self.modes),
            grid_size=len(self.weights),
            bytes_written=size,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PrecomputedSpectralKernel":
        """Load a persisted finite-spectral kernel."""

        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            modes = tuple(
                (int(ell), int(mode))
                for ell, mode in zip(data["ells"].tolist(), data["ms"].tolist())
            )
            return cls(
                lmax=int(data["lmax"]),
                n_theta=int(data["n_theta"]),
                n_phi=int(data["n_phi"]),
                transfer_scale=float(data["transfer_scale"]),
                modes=modes,
                weights=np.asarray(data["weights"]),
                source_values={
                    2: np.asarray(data["source_spin_plus"], dtype=complex),
                    -2: np.asarray(data["source_spin_minus"], dtype=complex),
                },
                path=source,
            )

    def response_on_source_grid(self, observable: str, vector, epsilon: int) -> np.ndarray:
        """Return one external response over the stored source grid."""

        source_spin = 2 * epsilon
        if source_spin not in self.source_values:
            raise ValueError("kernel does not contain the requested source spin")
        unit_vector = _unit(vector)
        key = (observable, int(epsilon), unit_vector.astype(np.float64).tobytes())
        if key not in self._response_cache:
            coefficients = _external_coefficients(
                unit_vector,
                observable,
                epsilon,
                self.modes,
                transfer_scale=self.transfer_scale,
            )
            self._response_cache[key] = coefficients @ self.source_values[source_spin]
        return self._response_cache[key]

    def evaluate(
        self,
        observables,
        vectors,
        epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    ) -> SpectralSumResult:
        """Evaluate one finite-spectral four-point kernel."""

        observables = _validate_observables(observables)
        epsilons = _validate_epsilons(epsilons)
        unit_vectors = tuple(_unit(vector) for vector in vectors)
        if len(unit_vectors) != 4:
            raise ValueError("four directions are required")
        product = np.ones(self.weights.shape, dtype=complex)
        for observable, vector, epsilon in zip(observables, unit_vectors, epsilons):
            product *= self.response_on_source_grid(observable, vector, epsilon)
        value = np.sum(self.weights * product) / (4.0 * math.pi)
        return SpectralSumResult(
            observables=observables,
            lmax=self.lmax,
            value=complex(value),
            n_theta=self.n_theta,
            n_phi=self.n_phi,
            backend="precomputed",
        )

    def evaluate_family(
        self,
        vectors,
        epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    ) -> dict[tuple[str, str, str, str], complex]:
        """Evaluate the standard PPPP-to-AAAA family with cached leg responses."""

        cases = (
            ("P", "P", "P", "P"),
            ("A", "P", "P", "P"),
            ("A", "A", "P", "P"),
            ("A", "A", "A", "P"),
            ("A", "A", "A", "A"),
        )
        epsilons = _validate_epsilons(epsilons)
        unit_vectors = tuple(_unit(vector) for vector in vectors)
        if len(unit_vectors) != 4:
            raise ValueError("four directions are required")
        leg_responses = {
            (leg_index, observable): self.response_on_source_grid(
                observable,
                unit_vectors[leg_index],
                epsilons[leg_index],
            )
            for leg_index in range(4)
            for observable in ("P", "A")
        }
        results = {}
        for case in cases:
            product = np.ones(self.weights.shape, dtype=complex)
            for leg_index, observable in enumerate(case):
                product *= leg_responses[(leg_index, observable)]
            results[case] = complex(np.sum(self.weights * product) / (4.0 * math.pi))
        return results


def default_precomputed_kernel_path(
    *,
    lmax: int = 15,
    n_theta: int | None = None,
    n_phi: int | None = None,
    directory: str | Path = DEFAULT_KERNEL_DIR,
) -> Path:
    """Return the default path for a persisted finite-spectral kernel."""

    n_theta = 2 * lmax + 8 if n_theta is None else int(n_theta)
    n_phi = 4 * lmax + 16 if n_phi is None else int(n_phi)
    return Path(directory) / f"fourpoint_spectral_lmax{lmax}_nt{n_theta}_np{n_phi}.npz"


def build_precomputed_spectral_kernel(
    *,
    lmax: int = 15,
    n_theta: int | None = None,
    n_phi: int | None = None,
    path: str | Path | None = None,
    transfer_scale: float = PHYSICAL_PTA_TRANSFER_SCALE,
) -> tuple[PrecomputedSpectralKernel, PrecomputedKernelBuildInfo]:
    """Build and persist a finite-spectral kernel."""

    kernel = PrecomputedSpectralKernel.build(
        lmax=lmax,
        n_theta=n_theta,
        n_phi=n_phi,
        transfer_scale=transfer_scale,
    )
    target = default_precomputed_kernel_path(lmax=lmax, n_theta=n_theta, n_phi=n_phi) if path is None else path
    info = kernel.save(target)
    return kernel, info


def load_precomputed_spectral_kernel(
    path: str | Path | None = None,
    *,
    lmax: int = 15,
    n_theta: int | None = None,
    n_phi: int | None = None,
    build_if_missing: bool = False,
) -> PrecomputedSpectralKernel:
    """Load a persisted kernel, optionally building it when absent."""

    source = default_precomputed_kernel_path(lmax=lmax, n_theta=n_theta, n_phi=n_phi) if path is None else Path(path)
    if not source.exists():
        if not build_if_missing:
            raise FileNotFoundError(source)
        kernel, _info = build_precomputed_spectral_kernel(
            lmax=lmax,
            n_theta=n_theta,
            n_phi=n_phi,
            path=source,
        )
        return kernel
    return PrecomputedSpectralKernel.load(source)
