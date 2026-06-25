"""Thin public calculator around the maintained four-point implementations."""

from __future__ import annotations

from dataclasses import dataclass

from celestial_intertwiners.fourpoint import (
    FastFourPointEvaluator,
    evaluate_kuntz_pppp_closed_form,
    load_precomputed_spectral_kernel,
)
from celestial_intertwiners.fourpoint.kuntz_benchmark import KUNTZ_SEED_EPSILONS


DEFAULT_FAMILY = (
    ("P", "P", "P", "P"),
    ("A", "P", "P", "P"),
    ("A", "A", "P", "P"),
    ("A", "A", "A", "P"),
    ("A", "A", "A", "A"),
)


@dataclass(frozen=True)
class AccuracyEntry:
    """One cached-kernel vs direct comparison."""

    observables: tuple[str, str, str, str]
    cached: complex
    direct: complex
    relative_error: float


class FourPointCalculator:
    """Unified access to Kuntz, direct, and precomputed mixed kernels."""

    def __init__(self, *, kernel=None, lmax: int = 15):
        self.kernel = kernel
        self.lmax = int(lmax)

    @classmethod
    def load_precomputed(cls, *, lmax: int = 15, path=None, build_if_missing: bool = False):
        """Load the persisted finite-spectral mixed-kernel surrogate."""

        kernel = load_precomputed_spectral_kernel(
            path,
            lmax=lmax,
            build_if_missing=build_if_missing,
        )
        return cls(kernel=kernel, lmax=kernel.lmax)

    def evaluate(
        self,
        observables,
        vectors,
        epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    ) -> complex:
        """Evaluate one cached finite-spectral mixed kernel."""

        if self.kernel is None:
            raise ValueError("load a precomputed kernel before calling evaluate")
        return self.kernel.evaluate(observables, vectors, epsilons).value

    def family(
        self,
        vectors,
        epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
    ) -> dict[tuple[str, str, str, str], complex]:
        """Evaluate the standard PPPP/APPP/AAPP/AAAP/AAAA family."""

        if self.kernel is None:
            raise ValueError("load a precomputed kernel before calling family")
        return self.kernel.evaluate_family(vectors, epsilons)

    def direct(
        self,
        observables,
        vectors,
        epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
        *,
        n_theta: int = 160,
        n_phi: int = 320,
        backend: str = "auto",
    ) -> complex:
        """Evaluate the physical antenna integral directly on CPU or CUDA."""

        return FastFourPointEvaluator(
            n_theta=n_theta,
            n_phi=n_phi,
            backend=backend,
        ).evaluate(observables, vectors, epsilons).value

    def direct_family(
        self,
        vectors,
        epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
        *,
        n_theta: int = 160,
        n_phi: int = 320,
        backend: str = "auto",
    ) -> dict[tuple[str, str, str, str], complex]:
        """Evaluate the standard family by direct antenna quadrature."""

        return FastFourPointEvaluator(
            n_theta=n_theta,
            n_phi=n_phi,
            backend=backend,
        ).evaluate_family(vectors, epsilons)

    def kuntz_pppp(self, vectors) -> complex:
        """Evaluate the Kuntz closed-form PPPP reference."""

        return evaluate_kuntz_pppp_closed_form(vectors)

    def compare(
        self,
        vectors,
        epsilons: tuple[int, int, int, int] = KUNTZ_SEED_EPSILONS,
        *,
        cases=DEFAULT_FAMILY,
        n_theta: int = 160,
        n_phi: int = 320,
        backend: str = "auto",
    ) -> tuple[AccuracyEntry, ...]:
        """Compare cached finite-spectral kernels against direct quadrature."""

        cached_family = self.family(vectors, epsilons)
        direct_family = self.direct_family(
            vectors,
            epsilons,
            n_theta=n_theta,
            n_phi=n_phi,
            backend=backend,
        )
        entries = []
        for case in cases:
            observables = tuple(case)
            cached = cached_family[observables]
            direct = direct_family[observables]
            relative_error = abs(cached - direct) / max(abs(direct), 1e-30)
            entries.append(
                AccuracyEntry(
                    observables=observables,
                    cached=cached,
                    direct=direct,
                    relative_error=float(relative_error),
                )
            )
        return tuple(entries)
