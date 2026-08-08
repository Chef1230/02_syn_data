"""Shared hard bounds for seconds-based time values.

Generation, task planning, and export all manipulate times as int64 Unix epoch
seconds.  The single source of truth for the representable range and the
database calendar interval lives here so every stage enforces the same bounds
instead of scattering ad-hoc checks.
"""

from __future__ import annotations

import numpy as np

# datetime64[ns] representable epoch-seconds range.  The extreme int64 value is
# the NaT sentinel, so the safe range is one second inside it on both ends
# (~1677-09-21T00:12:44Z .. 2262-04-11T23:47:16Z).
NS64_MIN_SECONDS = -9_223_372_036
NS64_MAX_SECONDS = 9_223_372_036


def seconds_in_datetime64_ns(seconds: np.ndarray) -> np.ndarray:
    """Boolean mask of values representable as a non-NaT datetime64[ns]."""
    return (seconds >= NS64_MIN_SECONDS) & (seconds <= NS64_MAX_SECONDS)


def assert_seconds_in_datetime64_ns(
    seconds: np.ndarray, *, context: str
) -> None:
    """Raise if any finite value cannot be represented as datetime64[ns]."""
    array = np.asarray(seconds)
    if array.size == 0:
        return
    numeric = array.astype(np.float64, copy=False)
    finite = numeric[np.isfinite(numeric)].astype(np.int64)
    bad = finite[~seconds_in_datetime64_ns(finite)]
    if bad.size:
        raise ValueError(
            f"{context}: timestamp seconds outside datetime64[ns] range "
            f"[{NS64_MIN_SECONDS}, {NS64_MAX_SECONDS}]: {int(bad[0])}"
        )


def assert_within_interval(
    values: np.ndarray, lo: int, hi: int, *, context: str
) -> None:
    """Raise if any value lies outside the inclusive interval ``[lo, hi]``."""
    array = np.asarray(values)
    if array.size == 0:
        return
    numeric = array.astype(np.float64, copy=False)
    if np.any(numeric < lo) or np.any(numeric > hi):
        raise ValueError(
            f"{context}: value outside [{lo}, {hi}]"
        )
