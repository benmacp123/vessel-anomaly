"""Scoring: matching planted events to detections, intervals and delays.

Injected (planted) events are the only US truth, so recall is measured on
planted positives and false alarms on planted negatives (near misses,
transits). A detection that overlaps no planted event is a real-data
alert, not a false positive: real alerts are reported as rates, never as
precision (PLAN.md, evaluation item 10).
"""

import math

import numpy as np
import polars as pl
from scipy.stats import chi2

# Two-sided 95% normal quantile for the Wilson interval.
Z95 = 1.959964


def wilson(successes: int, trials: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Args:
        successes: Count of successes.
        trials: Count of trials.
        z: Normal quantile for the confidence level.

    Returns:
        tuple[float, float]: Lower and upper bounds; (0, 1) with no trials.
    """
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = (z / (1 + z * z / trials)) * math.sqrt(
        p * (1 - p) / trials + z * z / (4 * trials * trials)
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def poisson_rate(
    count: int, exposure: float, per: float = 1.0, level: float = 0.95
) -> tuple[float, float, float]:
    """A rate with its exact (Garwood) Poisson interval.

    Args:
        count: Events observed.
        exposure: Exposure they were observed over (vessel-days, hours).
        per: Scale of the rate, for example 1,000 vessel-days.
        level: Confidence level.

    Returns:
        tuple[float, float, float]: Rate, lower and upper bound, each per
        ``per`` units of exposure.

    Raises:
        ValueError: If ``exposure`` is not positive.
    """
    if exposure <= 0:
        raise ValueError("exposure must be positive")
    alpha = 1 - level
    low = 0.0 if count == 0 else chi2.ppf(alpha / 2, 2 * count) / 2
    high = chi2.ppf(1 - alpha / 2, 2 * count + 2) / 2
    scale = per / exposure
    return count * scale, float(low) * scale, float(high) * scale


def match(
    truth: pl.DataFrame,
    found: pl.DataFrame,
    keys: list[str],
    slack_minutes: float = 0.0,
) -> pl.DataFrame:
    """Mark each planted event hit when a detection overlaps it.

    A detection hits a planted event when it shares every key and their
    intervals overlap, after widening the planted interval by
    ``slack_minutes`` each side. Several planted events may share one
    detection and the reverse; each planted event is scored once.

    Args:
        truth: Planted events with ``truth_id``, ``keys``, ``start`` and
            ``end``.
        found: Detections with ``keys``, ``start`` and ``end``.
        keys: Columns both must share (a vessel, a pair, a closure).
        slack_minutes: Tolerance on the interval ends.

    Returns:
        pl.DataFrame: ``truth`` plus ``hit`` (bool) and ``found_start``,
        the earliest overlapping detection's start (null when missed), in
        ``truth``'s row order.
    """
    slack = pl.duration(microseconds=round(slack_minutes * 60e6))
    pairs = (
        truth.select("truth_id", *keys, "start", "end")
        .join(
            found.select(*keys, found_start="start", found_end="end"),
            on=keys,
            how="inner",
        )
        .filter(
            (pl.col("found_start") <= pl.col("end") + slack)
            & (pl.col("found_end") >= pl.col("start") - slack)
        )
        .group_by("truth_id")
        .agg(pl.col("found_start").min())
    )
    return truth.join(
        pairs, on="truth_id", how="left", maintain_order="left"
    ).with_columns(hit=pl.col("found_start").is_not_null())


def confusion(positives: pl.Series, negatives: pl.Series) -> dict[str, int]:
    """Confusion counts from planted positives and negatives.

    Args:
        positives: ``hit`` for each planted positive.
        negatives: ``hit`` for each planted negative.

    Returns:
        dict[str, int]: ``tp``, ``fn``, ``fp`` and ``tn``.
    """
    tp = int(positives.sum())
    fp = int(negatives.sum())
    return {"tp": tp, "fn": positives.len() - tp, "fp": fp, "tn": negatives.len() - fp}


def scores(counts: dict[str, int], beta: float = 0.5) -> dict[str, float | None]:
    """Recall, precision and F-beta on planted events only.

    Precision here depends on how many negatives were planted, so it is
    reported with that count and never compared with real-data rates.

    Args:
        counts: ``confusion`` output.
        beta: F-score weight; 0.5 favours precision, as Welch did.

    Returns:
        dict[str, float | None]: ``recall``, ``precision``, ``f_beta`` and
        ``false_alarm`` (share of negatives hit); None where undefined.
    """
    tp, fn, fp, tn = counts["tp"], counts["fn"], counts["fp"], counts["tn"]
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    if recall is None or precision is None or recall + precision == 0:
        f_beta = None
    else:
        b2 = beta * beta
        f_beta = (1 + b2) * precision * recall / (b2 * precision + recall)
    false_alarm = fp / (fp + tn) if fp + tn else None
    return {
        "recall": recall,
        "precision": precision,
        "f_beta": f_beta,
        "false_alarm": false_alarm,
    }


def delay_summary(hours: pl.Series) -> dict[str, float | None]:
    """Median and 90th percentile of detection delays.

    Args:
        hours: Delays in hours; nulls (never alerted) are left out.

    Returns:
        dict[str, float | None]: ``n``, ``median_hours`` and
        ``p90_hours``; the two quantiles are None with no delays.
    """
    values = hours.drop_nulls().to_numpy()
    if values.size == 0:
        return {"n": 0, "median_hours": None, "p90_hours": None}
    return {
        "n": int(values.size),
        "median_hours": float(np.median(values)),
        "p90_hours": float(np.quantile(values, 0.9)),
    }
