# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Numba-friendly ephemeris helpers that return raw arrays."""

import erfa

from astropy.coordinates import solar_system as _solar_system
from astropy.coordinates.solar_system import solar_system_ephemeris

from . import units as _units


def get_body_barycentric_posvel_jd(body, jd1, jd2, ephemeris=None, unit="au"):
    """Return barycentric position/velocity arrays for TDB JD inputs.

    Parameters
    ----------
    body : str or list of tuple
        The solar system body for which to calculate positions.  Can also be a
        kernel specifier (list of 2-tuples) if the ``ephemeris`` is a JPL kernel.
    jd1, jd2 : float or array-like
        TDB Julian date split into two parts.
    ephemeris : str, optional
        Ephemeris to use.  By default, use the one set with
        ``astropy.coordinates.solar_system_ephemeris.set``.
    unit : {'au', 'km'}
        Unit to return for the arrays. Velocities are returned per day.
    """
    pos, vel, native_unit = _solar_system._get_body_barycentric_posvel_arrays(
        body, jd1, jd2, ephemeris=ephemeris, get_velocity=True
    )
    if native_unit == unit:
        return pos, vel

    if native_unit == "km" and unit == "au":
        return _units.km_to_au(pos), _units.km_per_day_to_au_per_day(vel)
    if native_unit == "au" and unit == "km":
        return _units.au_to_km(pos), _units.au_per_day_to_km_per_day(vel)

    raise ValueError(f"Unsupported unit '{unit}' (expected 'au' or 'km').")


def prepare_earth_position_vel_jd(jd1, jd2, ephemeris=None, unit="au"):
    """Return Earth barycentric position/velocity and heliocentric position arrays.

    Parameters
    ----------
    jd1, jd2 : float or array-like
        TDB Julian date split into two parts.
    ephemeris : str, optional
        Ephemeris to use.  By default, use the one set with
        ``astropy.coordinates.solar_system_ephemeris.set``.
    unit : {'au', 'km'}
        Unit to return for the arrays. Velocities are returned per day.

    Returns
    -------
    earth_pos : `~numpy.ndarray`
        Barycentric position of Earth, shape ``(..., 3)``.
    earth_vel : `~numpy.ndarray`
        Barycentric velocity of Earth, shape ``(..., 3)``.
    earth_heliocentric : `~numpy.ndarray`
        Heliocentric position of Earth, shape ``(..., 3)``.
    """
    if ephemeris is None:
        ephemeris = solar_system_ephemeris.get()

    if ephemeris == "builtin":
        earth_pv_heliocentric, earth_pv = erfa.epv00(jd1, jd2)
        earth_pos = earth_pv["p"]
        earth_vel = earth_pv["v"]
        earth_heliocentric = earth_pv_heliocentric["p"]
        native_unit = "au"
    else:
        earth_pos, earth_vel, native_unit = (
            _solar_system._get_body_barycentric_posvel_arrays(
                "earth", jd1, jd2, ephemeris=ephemeris, get_velocity=True
            )
        )
        sun_pos, _, sun_unit = _solar_system._get_body_barycentric_posvel_arrays(
            "sun", jd1, jd2, ephemeris=ephemeris, get_velocity=False
        )
        if sun_unit != native_unit:
            if sun_unit == "km":
                sun_pos = _units.km_to_au(sun_pos)
                sun_unit = "au"
            else:
                sun_pos = _units.au_to_km(sun_pos)
                sun_unit = "km"
        earth_heliocentric = earth_pos - sun_pos

    if unit == native_unit:
        return earth_pos, earth_vel, earth_heliocentric
    if native_unit == "km" and unit == "au":
        return (
            _units.km_to_au(earth_pos),
            _units.km_per_day_to_au_per_day(earth_vel),
            _units.km_to_au(earth_heliocentric),
        )
    if native_unit == "au" and unit == "km":
        return (
            _units.au_to_km(earth_pos),
            _units.au_per_day_to_km_per_day(earth_vel),
            _units.au_to_km(earth_heliocentric),
        )

    raise ValueError(f"Unsupported unit '{unit}' (expected 'au' or 'km').")


__all__ = ["get_body_barycentric_posvel_jd", "prepare_earth_position_vel_jd"]
