"""Aggregation across samples and across trials.

The trial statistics are the part worth reading carefully. Benchmarks with
stochastic scenarios do not produce a number, they produce a *distribution*,
and a single run is not a measurement of it. Reporting mean ± a normal-based
interval over three trials would be wrong twice: the sample is tiny, and the
normal approximation understates the spread at that size. So the interval is
computed from Student's t.

No scipy: a table of two-tailed t critical values covers the small degrees of
freedom that actually occur, and the normal quantile takes over once the two
agree to three decimal places.
"""

from __future__ import annotations

import logging
import statistics
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Two-tailed Student's t critical values, indexed by degrees of freedom.
# Beyond df=30 the normal quantile is within ~0.5%, so it is used instead.
_T_TABLE: Dict[float, Dict[int, float]] = {
    0.90: {
        1: 6.314,
        2: 2.920,
        3: 2.353,
        4: 2.132,
        5: 2.015,
        6: 1.943,
        7: 1.895,
        8: 1.860,
        9: 1.833,
        10: 1.812,
        11: 1.796,
        12: 1.782,
        13: 1.771,
        14: 1.761,
        15: 1.753,
        16: 1.746,
        17: 1.740,
        18: 1.734,
        19: 1.729,
        20: 1.725,
        21: 1.721,
        22: 1.717,
        23: 1.714,
        24: 1.711,
        25: 1.708,
        26: 1.706,
        27: 1.703,
        28: 1.701,
        29: 1.699,
        30: 1.697,
    },
    0.95: {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    },
    0.99: {
        1: 63.657,
        2: 9.925,
        3: 5.841,
        4: 4.604,
        5: 4.032,
        6: 3.707,
        7: 3.499,
        8: 3.355,
        9: 3.250,
        10: 3.169,
        11: 3.106,
        12: 3.055,
        13: 3.012,
        14: 2.977,
        15: 2.947,
        16: 2.921,
        17: 2.898,
        18: 2.878,
        19: 2.861,
        20: 2.845,
        21: 2.831,
        22: 2.819,
        23: 2.807,
        24: 2.797,
        25: 2.787,
        26: 2.779,
        27: 2.771,
        28: 2.763,
        29: 2.756,
        30: 2.750,
    },
}

_MAX_TABULATED_DF = 30


def t_critical(df: int, level: float = 0.95) -> float:
    """Return the two-tailed Student's t critical value for ``df``.

    Falls back to the normal quantile beyond the tabulated range, and for
    confidence levels other than 0.90 / 0.95 / 0.99.
    """
    if df < 1:
        return 0.0
    table = _T_TABLE.get(round(level, 2))
    if table is None:
        logger.debug(
            "no t table for confidence level %.3f; using the normal quantile", level
        )
        return statistics.NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)
    if df <= _MAX_TABULATED_DF:
        return table[df]
    return statistics.NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)


def mean_with_ci(
    values: Sequence[float], level: float = 0.95
) -> tuple[float, float, float, float]:
    """Return ``(mean, stdev, ci_low, ci_high)`` for ``values``.

    A single value has no spread to estimate, so its interval is the point
    itself rather than a fabricated range.
    """
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0, mean, mean

    stdev = statistics.stdev(values)
    halfwidth = t_critical(len(values) - 1, level) * stdev / (len(values) ** 0.5)
    return mean, stdev, mean - halfwidth, mean + halfwidth


def mean_sub_scores(dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Average each named sub-score across trials, keyed by name.

    A sub-score missing from some trials is averaged over the trials that
    reported it, rather than being treated as zero.
    """
    buckets: Dict[str, List[float]] = {}
    for item in dicts:
        for key, value in (item or {}).items():
            buckets.setdefault(key, []).append(float(value))
    return {key: statistics.fmean(vals) for key, vals in buckets.items() if vals}


def unstable_sub_scores(
    dicts: Sequence[Dict[str, float]], *, tolerance: float = 1e-9
) -> List[str]:
    """Return the sub-scores that disagreed across trials.

    These are the stochastic parts of a benchmark -- the ones worth re-running
    when targeted re-trials are enabled, instead of paying to repeat the suite.
    """
    if len(dicts) < 2:
        return []
    buckets: Dict[str, List[float]] = {}
    for item in dicts:
        for key, value in (item or {}).items():
            buckets.setdefault(key, []).append(float(value))
    return sorted(
        key
        for key, vals in buckets.items()
        if len(vals) > 1 and (max(vals) - min(vals)) > tolerance
    )


def score_population(
    values: Sequence[float],
    n_errors: int,
    failure_tolerance: float,
    invalid_score: float,
) -> Optional[float]:
    """Mean of ``values``, or ``invalid_score`` if too much of it errored.

    Scoring a benchmark on the records that happened to succeed silently
    changes what was measured, so past the tolerance the result is reported as
    invalid rather than optimistic.
    """
    total = len(values) + n_errors
    if total == 0:
        return invalid_score
    if failure_tolerance >= 0 and (n_errors / total) > failure_tolerance:
        logger.warning(
            "%d/%d records errored (>%.0f%% tolerance); reporting an invalid score",
            n_errors,
            total,
            failure_tolerance * 100,
        )
        return invalid_score
    if not values:
        return invalid_score
    return statistics.fmean(values)


__all__ = [
    "mean_sub_scores",
    "mean_with_ci",
    "score_population",
    "t_critical",
    "unstable_sub_scores",
]
