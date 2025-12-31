# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Numba-friendly helpers for Time and TimeDelta extraction."""

from astropy.coordinates.builtin_frames.utils import get_jd12


def time_to_jd1_jd2(time, scale="tdb"):
    """Return split JD values (jd1, jd2) in the requested scale."""
    return get_jd12(time, scale)


__all__ = ["time_to_jd1_jd2"]
