# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Numba-oriented helpers for performance-critical numeric paths."""

try:
    import numba as _numba
except Exception:  # pragma: no cover - optional dependency
    _numba = None

HAS_NUMBA = _numba is not None


def _maybe_njit(func):
    if _numba is None:
        return None
    return _numba.njit(cache=True)(func)


def njit_nogil(*args, **kwargs):
    """Return a numba.njit decorator with nogil enabled by default."""
    if _numba is None:
        raise ModuleNotFoundError("numba is not installed")
    kwargs.setdefault("nogil", True)
    kwargs.setdefault("cache", True)
    return _numba.njit(*args, **kwargs)


__all__ = ["HAS_NUMBA", "njit_nogil"]
