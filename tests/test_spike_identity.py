"""Tests for the identity-match spike: the VMS cut (box buffer and
de-duplication) and the VMS-to-AIS fix matching."""

import importlib.util
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "spike_identity.py"
T0 = datetime(2024, 5, 27, 12, 0, tzinfo=timezone.utc)
# Same sphere as the script, so test distances are exact on its scale.
EARTH_RADIUS_M = 6_371_008.8


@pytest.fixture(name="spike")
def fixture_spike():
    """Load the spike script as a module, since ``scripts/`` is not a package.

    Returns:
        module: The loaded spike script.
    """
    spec = importlib.util.spec_from_file_location("spike_identity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixes(rows):
    """A minimal VMS frame.

    Args:
        rows: Tuples of (message_id, call_sign, lon, lat).

    Returns:
        pl.DataFrame: Fixes all at ``T0``.
    """
    ids, calls, lons, lats = zip(*rows)
    return pl.DataFrame(
        {
            "message_id": ids,
            "call_sign": calls,
            "lon": lons,
            "lat": lats,
            "timestamp": [T0] * len(rows),
        }
    )


def test_buffer_keeps_near_and_drops_far(spike):
    """A fix 5 km south of Sula is kept; one 20 km south is not."""
    sula_south = 63.65
    near = sula_south - 5 / spike.KM_PER_DEGREE_LAT
    far = sula_south - 20 / spike.KM_PER_DEGREE_LAT
    cut = spike.cut_vms(fixes([(1, "A", 8.0, near), (2, "B", 8.0, far)]))
    assert cut["call_sign"].to_list() == ["A"]
    assert cut["box"].to_list() == ["sula"]


def test_same_minute_repeats_keep_lowest_id(spike):
    """Two reports of one vessel in one minute collapse to the lower ID."""
    cut = spike.cut_vms(fixes([(9, "A", 8.0, 64.0), (4, "A", 8.0001, 64.0)]))
    assert cut["message_id"].to_list() == [4]


def test_overlap_takes_first_listed_box(spike):
    """A fix in the buffered Røst-Træna overlap is labelled Røst."""
    cut = spike.cut_vms(fixes([(1, "A", 10.3, 67.1)]))
    assert cut["box"].to_list() == ["rost"]


def test_cut_flags_unbuffered_box(spike):
    """Fixes in the buffer are kept but not flagged as in the box."""
    near = 63.65 - 5 / spike.KM_PER_DEGREE_LAT
    cut = spike.cut_vms(fixes([(1, "A", 8.0, near), (2, "B", 8.0, 64.0)]))
    assert dict(zip(cut["call_sign"], cut["in_box"])) == {"A": False, "B": True}


def metres_north(lat, metres):
    """Shift a latitude north by a distance.

    Args:
        lat: Latitude in degrees.
        metres: Distance north.

    Returns:
        float: The new latitude.
    """
    return lat + math.degrees(metres / EARTH_RADIUS_M)


def one_vms_fix():
    """A single in-box VMS fix at ``T0``, 8E 64N.

    Returns:
        pl.DataFrame: The fix, shaped like ``cut_vms`` output.
    """
    return pl.DataFrame(
        {
            "message_id": [7],
            "call_sign": ["LAAA"],
            "in_box": [True],
            "lon": [8.0],
            "lat": [64.0],
            "timestamp": [T0],
        }
    )


def ais(rows):
    """A minimal AIS frame, sorted by time.

    Args:
        rows: Tuples of (mmsi, seconds after ``T0``, metres north of 64N).

    Returns:
        pl.DataFrame: Reports at 8E.
    """
    mmsi, offsets, north = zip(*rows)
    frame = pl.DataFrame(
        {
            "mmsi": mmsi,
            "timestamp": [T0 + timedelta(seconds=s) for s in offsets],
            "lon": [8.0] * len(rows),
            "lat": [metres_north(64.0, m) for m in north],
        }
    )
    return frame.sort("timestamp")


def test_match_needs_both_space_and_time(spike):
    """Only the MMSI close in both distance and time matches."""
    reports = ais(
        [
            (1, 30, 100),  # close in both: matches
            (2, 30, 600),  # too far
            (3, 150, 100),  # too late
            (4, -119, 499),  # just inside both limits
        ]
    )
    matches = spike.match_fixes(one_vms_fix(), reports)
    assert sorted(matches["mmsi"].to_list()) == [1, 4]


def test_match_keeps_closest_per_mmsi(spike):
    """Several AIS fixes from one MMSI give one row with the best values."""
    reports = ais([(1, -60, 300), (1, 10, 50), (1, 90, 200)])
    matches = spike.match_fixes(one_vms_fix(), reports)
    assert matches.height == 1
    assert matches["distance_m"][0] == pytest.approx(50, abs=0.1)
    assert matches["dt_s"][0] == 10


def test_shift_moves_vms_time(spike):
    """A 3 h shift matches AIS 3 h later and loses the original match."""
    reports = ais([(1, 0, 0), (2, 3 * 3600, 0)])
    matches = spike.match_fixes(one_vms_fix(), reports, shift=timedelta(hours=3))
    assert matches["mmsi"].to_list() == [2]


def test_pair_counts_count_vms_fixes(spike):
    """Pairs count VMS fixes, however many AIS fixes matched each one."""
    second = one_vms_fix().with_columns(message_id=pl.lit(8, dtype=pl.Int64))
    vms = pl.concat([one_vms_fix(), second])
    matches = pl.DataFrame(
        {
            "message_id": [7, 8],
            "call_sign": ["LAAA", "LAAA"],
            "in_box": [True, True],
            "mmsi": [1, 1],
            "distance_m": [10.0, 20.0],
            "dt_s": [5, 6],
        }
    )
    pairs = spike.pair_counts(vms, matches)
    assert pairs.select("matched", "vms_fixes").row(0) == (2, 2)


def vms_counts(counts):
    """VMS fixes, all in a box, with a given number per call sign.

    Args:
        counts: Call sign to number of in-box fixes.

    Returns:
        pl.DataFrame: Fixes with ``call_sign`` and ``in_box`` only.
    """
    calls = [call for call, n in counts.items() for _ in range(n)]
    return pl.DataFrame({"call_sign": calls, "in_box": [True] * len(calls)})


def pairs_table(rows):
    """A pairs frame with the columns ``assign`` reads.

    Args:
        rows: Tuples of (call_sign, mmsi, matched_in_box).

    Returns:
        pl.DataFrame: The pairs.
    """
    calls, mmsi, matched = zip(*rows)
    return pl.DataFrame(
        {"call_sign": calls, "mmsi": mmsi, "matched_in_box": matched},
        schema={"call_sign": pl.String, "mmsi": pl.Int64, "matched_in_box": pl.UInt32},
    )


def status_of(spike, counts, rows):
    """Run ``assign`` and map each call sign to its status.

    Args:
        spike: The loaded spike module.
        counts: Call sign to number of in-box VMS fixes.
        rows: Pairs as for ``pairs_table``.

    Returns:
        dict: Call sign to status.
    """
    table = spike.assign(vms_counts(counts), pairs_table(rows))
    return dict(zip(table["call_sign"], table["status"]))


def test_assign_clear_leader(spike):
    """A leader with enough fixes, share and margin is assigned."""
    assert status_of(spike, {"A": 10}, [("A", 1, 9), ("A", 2, 4)]) == {"A": "assigned"}


def test_assign_rejects_low_share_and_few_fixes(spike):
    """Under 80% of fixes, or under 5 matched fixes, is unmatched."""
    statuses = status_of(spike, {"A": 10, "B": 5}, [("A", 1, 7), ("B", 2, 4)])
    assert statuses == {"A": "unmatched", "B": "unmatched"}


def test_assign_close_runner_up_is_ambiguous(spike):
    """A runner-up with more than half the leader's fixes blocks assignment."""
    assert status_of(spike, {"A": 10}, [("A", 1, 10), ("A", 2, 6)]) == {
        "A": "ambiguous"
    }


def test_assign_shared_leader_is_ambiguous(spike):
    """One MMSI leading two call signs is assigned to neither."""
    statuses = status_of(spike, {"A": 10, "B": 10}, [("A", 1, 10), ("B", 1, 9)])
    assert statuses == {"A": "ambiguous", "B": "ambiguous"}


def test_assign_counts_unmatched_and_skips_small(spike):
    """A call sign with no AIS match counts; one under 5 fixes is not eligible."""
    statuses = status_of(spike, {"A": 6, "B": 4}, [("B", 1, 4)])
    assert statuses == {"A": "unmatched"}


def test_wilson_interval_known_value(spike):
    """7 of 10 gives the textbook Wilson 95% interval."""
    low, high = spike.wilson_interval(7, 10)
    assert (low, high) == (
        pytest.approx(0.3968, abs=1e-4),
        pytest.approx(0.8922, abs=1e-4),
    )


def test_compare_control_separates_same_and_different(spike):
    """Control assignments are split into the real MMSI and strangers."""
    real = pl.DataFrame(
        {
            "call_sign": ["A", "B", "C"],
            "mmsi": [1, 2, None],
            "status": ["assigned"] * 2 + ["unmatched"],
        },
        schema_overrides={"mmsi": pl.Int64},
    )
    control = pl.DataFrame(
        {
            "call_sign": ["A", "B", "C"],
            "mmsi": [1, 9, 7],
            "status": ["assigned", "assigned", "assigned"],
        },
    )
    compared = spike.compare_control(real, control)
    assert dict(zip(compared["call_sign"], compared["same_mmsi"])) == {
        "A": True,
        "B": False,
        "C": False,
    }
