# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel
from astropy.coordinates.builtin_frames.utils import prepare_earth_position_vel
from astropy.numba import solar_system as nbss
from astropy.numba import time as nbtime
from astropy.time import Time


def test_get_body_barycentric_posvel_jd_units():
    time = Time("2024-01-01T00:00:00", scale="tdb")
    jd1, jd2 = nbtime.time_to_jd1_jd2(time, "tdb")

    pos_raw, vel_raw = nbss.get_body_barycentric_posvel_jd(
        "earth", jd1, jd2, ephemeris="builtin", unit="au"
    )
    pos, vel = get_body_barycentric_posvel("earth", time, ephemeris="builtin")

    np.testing.assert_allclose(pos_raw, pos.get_xyz(xyz_axis=-1).to_value(u.au))
    np.testing.assert_allclose(
        vel_raw, vel.get_xyz(xyz_axis=-1).to_value(u.au / u.day)
    )

    pos_km, vel_km = nbss.get_body_barycentric_posvel_jd(
        "earth", jd1, jd2, ephemeris="builtin", unit="km"
    )
    np.testing.assert_allclose(pos_km, pos.get_xyz(xyz_axis=-1).to_value(u.km))
    np.testing.assert_allclose(
        vel_km, vel.get_xyz(xyz_axis=-1).to_value(u.km / u.day)
    )


def test_prepare_earth_position_vel_jd_builtin():
    time = Time("2024-01-01T00:00:00", scale="tdb")
    jd1, jd2 = nbtime.time_to_jd1_jd2(time, "tdb")
    earth_pv, earth_helio = prepare_earth_position_vel(time)

    pos, vel, helio = nbss.prepare_earth_position_vel_jd(
        jd1, jd2, ephemeris="builtin", unit="au"
    )
    np.testing.assert_allclose(pos, earth_pv["p"])
    np.testing.assert_allclose(vel, earth_pv["v"])
    np.testing.assert_allclose(helio, earth_helio)
