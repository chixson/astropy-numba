# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Numba-friendly unit conversions for common heliocentric workflows."""

import numpy as np

from astropy.constants import si as _si

from . import _maybe_njit

AU_KM = _si.au.value / 1000.0
DAY_S = 86400.0


def au_to_km(value):
    return np.asarray(value) * AU_KM


def km_to_au(value):
    return np.asarray(value) / AU_KM


def au_per_day_to_km_per_day(value):
    return np.asarray(value) * AU_KM


def km_per_day_to_au_per_day(value):
    return np.asarray(value) / AU_KM


def km_per_s_to_au_per_day(value):
    return np.asarray(value) * (DAY_S / AU_KM)


def au_per_day_to_km_per_s(value):
    return np.asarray(value) * (AU_KM / DAY_S)


au_to_km_jit = _maybe_njit(au_to_km)
km_to_au_jit = _maybe_njit(km_to_au)
au_per_day_to_km_per_day_jit = _maybe_njit(au_per_day_to_km_per_day)
km_per_day_to_au_per_day_jit = _maybe_njit(km_per_day_to_au_per_day)
km_per_s_to_au_per_day_jit = _maybe_njit(km_per_s_to_au_per_day)
au_per_day_to_km_per_s_jit = _maybe_njit(au_per_day_to_km_per_s)


__all__ = [
    "AU_KM",
    "DAY_S",
    "au_to_km",
    "km_to_au",
    "au_per_day_to_km_per_day",
    "km_per_day_to_au_per_day",
    "km_per_s_to_au_per_day",
    "au_per_day_to_km_per_s",
    "au_to_km_jit",
    "km_to_au_jit",
    "au_per_day_to_km_per_day_jit",
    "km_per_day_to_au_per_day_jit",
    "km_per_s_to_au_per_day_jit",
    "au_per_day_to_km_per_s_jit",
]
