"""Tensor-GR PTA and astrometric transfer functions."""

from __future__ import annotations

import math

import sympy as sp


def _validate_l(l: int, allow_zero: bool = False) -> bool:
    if int(l) != l:
        raise ValueError("l must be an integer")
    if l < 2:
        if allow_zero:
            return False
        raise ValueError("tensor transfer functions require l >= 2")
    return True


def pta_transfer(l: int, *, allow_zero: bool = False) -> float:
    """Return p_l = 2 / sqrt((l-1) l (l+1) (l+2))."""

    if not _validate_l(l, allow_zero=allow_zero):
        return 0.0
    return 2.0 / math.sqrt((l - 1) * l * (l + 1) * (l + 2))


def pta_transfer_exact(l: int, *, allow_zero: bool = False) -> sp.Expr:
    """Exact SymPy version of :func:`pta_transfer`."""

    if not _validate_l(l, allow_zero=allow_zero):
        return sp.S.Zero
    l_sym = sp.Integer(l)
    return sp.Integer(2) / sp.sqrt((l_sym - 1) * l_sym * (l_sym + 1) * (l_sym + 2))


def astrometry_transfer(l: int, *, allow_zero: bool = False) -> float:
    """Return t_l = 2 / (l(l+1) sqrt((l-1)(l+2)))."""

    if not _validate_l(l, allow_zero=allow_zero):
        return 0.0
    return 2.0 / (l * (l + 1) * math.sqrt((l - 1) * (l + 2)))


def astrometry_transfer_exact(l: int, *, allow_zero: bool = False) -> sp.Expr:
    """Exact SymPy version of :func:`astrometry_transfer`."""

    if not _validate_l(l, allow_zero=allow_zero):
        return sp.S.Zero
    l_sym = sp.Integer(l)
    return sp.Integer(2) / (
        l_sym * (l_sym + 1) * sp.sqrt((l_sym - 1) * (l_sym + 2))
    )


def astrometry_to_pta_ratio(l: int, *, allow_zero: bool = False) -> float:
    """Return t_l / p_l = 1 / sqrt(l(l+1))."""

    if not _validate_l(l, allow_zero=allow_zero):
        return 0.0
    return 1.0 / math.sqrt(l * (l + 1))


def astrometry_to_pta_ratio_exact(l: int, *, allow_zero: bool = False) -> sp.Expr:
    """Exact SymPy version of :func:`astrometry_to_pta_ratio`."""

    if not _validate_l(l, allow_zero=allow_zero):
        return sp.S.Zero
    l_sym = sp.Integer(l)
    return sp.Integer(1) / sp.sqrt(l_sym * (l_sym + 1))


def astrometry_cl_weight(l: int, *, allow_zero: bool = False) -> float:
    """Return a_l = (2l+1) t_l**2."""

    if not _validate_l(l, allow_zero=allow_zero):
        return 0.0
    return (2 * l + 1) * astrometry_transfer(l) ** 2


def astrometry_cl_weight_exact(l: int, *, allow_zero: bool = False) -> sp.Expr:
    """Return exact a_l = 4(2l+1)/((l-1)l^2(l+1)^2(l+2))."""

    if not _validate_l(l, allow_zero=allow_zero):
        return sp.S.Zero
    l_sym = sp.Integer(l)
    return sp.Integer(4) * (2 * l_sym + 1) / (
        (l_sym - 1) * l_sym**2 * (l_sym + 1) ** 2 * (l_sym + 2)
    )
