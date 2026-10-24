"""Tests for the scoring code against hand-built cases and published
interval values."""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from vessels.evaluation.metrics import confusion, delay_summary, match
from vessels.evaluation.metrics import poisson_rate, scores, wilson

T0 = datetime(2024, 5, 20, tzinfo=timezone.utc)


def at(hours: float) -> datetime:
    """A time some hours after T0.

    Args:
        hours: Offset in hours.

    Returns:
        datetime: The time.
    """
    return T0 + timedelta(hours=hours)


def frame(rows: list[tuple], columns: list[str]) -> pl.DataFrame:
    """A small frame with microsecond UTC times.

    Args:
        rows: Row tuples.
        columns: Column names.

    Returns:
        pl.DataFrame: The frame.
    """
    data = pl.DataFrame(rows, schema=columns, orient="row")
    return data.with_columns(pl.col("start", "end").cast(pl.Datetime("us", "UTC")))


def test_wilson_matches_published_values():
    """Wilson bounds for 8 of 10 match the textbook 0.490-0.943."""
    low, high = wilson(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-4)
    assert high == pytest.approx(0.9433, abs=1e-4)


def test_wilson_edges():
    """All-success and no-trial cases stay inside [0, 1]."""
    assert wilson(10, 10)[1] == pytest.approx(1.0)
    assert wilson(0, 10)[0] == pytest.approx(0.0)
    assert wilson(0, 0) == (0.0, 1.0)


def test_poisson_matches_published_values():
    """Garwood bounds for 0 and 5 events match published tables."""
    assert poisson_rate(0, 1.0) == pytest.approx((0.0, 0.0, 3.6889), abs=1e-4)
    rate, low, high = poisson_rate(5, 1.0)
    assert (rate, low, high) == pytest.approx((5.0, 1.6235, 11.6683), abs=1e-4)


def test_poisson_scales_per_exposure():
    """Rates scale to 'per 1,000' of the exposure unit."""
    rate, low, high = poisson_rate(5, 2_000.0, per=1_000.0)
    assert rate == pytest.approx(2.5)
    assert low == pytest.approx(1.6235 / 2, abs=1e-4)
    assert high == pytest.approx(11.6683 / 2, abs=1e-4)


def test_poisson_rejects_zero_exposure():
    """A rate over no exposure is an error, not infinity."""
    with pytest.raises(ValueError):
        poisson_rate(1, 0.0)


def test_hand_built_confusion_matrix():
    """Seven positives (5 hit) and four negatives (1 hit) score as by hand."""
    counts = confusion(
        pl.Series([True, True, False, True, True, False, True]),
        pl.Series([False, True, False, False]),
    )
    assert counts == {"tp": 5, "fn": 2, "fp": 1, "tn": 3}
    result = scores(counts)
    assert result["recall"] == pytest.approx(5 / 7)
    assert result["precision"] == pytest.approx(5 / 6)
    # F0.5 = 1.25 P R / (0.25 P + R) with P = 5/6, R = 5/7.
    assert result["f_beta"] == pytest.approx(
        1.25 * (5 / 6) * (5 / 7) / (0.25 * (5 / 6) + 5 / 7)
    )
    assert result["false_alarm"] == pytest.approx(1 / 4)


def test_scores_undefined_without_cases():
    """With nothing planted, every score is undefined rather than zero."""
    result = scores({"tp": 0, "fn": 0, "fp": 0, "tn": 0})
    assert result == {
        "recall": None,
        "precision": None,
        "f_beta": None,
        "false_alarm": None,
    }


def test_match_needs_same_key_and_overlap():
    """Only a detection of the same vessel overlapping in time hits."""
    truth = frame(
        [
            ("t1", "a", at(0), at(2)),
            ("t2", "a", at(10), at(12)),
            ("t3", "b", at(0), at(2)),
            ("t4", "c", at(0), at(2)),
        ],
        ["truth_id", "vessel_id", "start", "end"],
    )
    found = frame(
        [
            ("a", at(1), at(3)),
            ("a", at(5), at(6)),
            ("c", at(3), at(4)),
            ("b", at(1.5), at(1.6)),
            ("b", at(0.5), at(0.7)),
        ],
        ["vessel_id", "start", "end"],
    )
    result = match(truth, found, ["vessel_id"])
    assert result["truth_id"].to_list() == ["t1", "t2", "t3", "t4"]
    assert result["hit"].to_list() == [True, False, True, False]
    assert result["found_start"][2] == at(0.5)


def test_match_slack_widens_interval():
    """A detection just after the planted end hits only with slack."""
    truth = frame(
        [("t1", "a", at(0), at(2))], ["truth_id", "vessel_id", "start", "end"]
    )
    found = frame([("a", at(2.25), at(3))], ["vessel_id", "start", "end"])
    assert not match(truth, found, ["vessel_id"])["hit"][0]
    assert match(truth, found, ["vessel_id"], slack_minutes=30)["hit"][0]


def test_match_with_no_detections():
    """No detections means every planted event is missed."""
    truth = frame(
        [("t1", "a", at(0), at(2))], ["truth_id", "vessel_id", "start", "end"]
    )
    found = frame([], ["vessel_id", "start", "end"]).cast({"vessel_id": pl.String})
    assert match(truth, found, ["vessel_id"])["hit"].to_list() == [False]


def test_delay_summary():
    """Median and 90th percentile ignore events never alerted."""
    result = delay_summary(pl.Series([1.0, 2.0, 3.0, None, 4.0, 5.0]))
    assert result["n"] == 5
    assert result["median_hours"] == 3.0
    assert result["p90_hours"] == pytest.approx(4.6)
    assert delay_summary(pl.Series([None], dtype=pl.Float64))["median_hours"] is None
