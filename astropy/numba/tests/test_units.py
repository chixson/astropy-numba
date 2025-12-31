# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from astropy.numba import units as nbu


def test_basic_unit_conversions():
    values = np.array([0.0, 1.0, 2.5])
    km = nbu.au_to_km(values)
    au = nbu.km_to_au(km)
    np.testing.assert_allclose(au, values)

    kms = np.array([0.0, 1.0, 12.3])
    au_per_day = nbu.km_per_s_to_au_per_day(kms)
    km_per_s = nbu.au_per_day_to_km_per_s(au_per_day)
    np.testing.assert_allclose(km_per_s, kms)
