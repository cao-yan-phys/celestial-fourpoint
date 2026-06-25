"""Connected four-point cumulant normalization helpers."""

from __future__ import annotations

from .conventions import temporal_transfer


def temporal_prefactor(observables, frequencies, *, exact=False):
    """Return product_i g_{X_i}(f_i)."""

    if len(observables) != len(frequencies):
        raise ValueError("observables and frequencies must have the same length")
    factor = 1
    for observable, frequency in zip(observables, frequencies):
        factor *= temporal_transfer(observable, frequency, exact=exact)
    return factor
