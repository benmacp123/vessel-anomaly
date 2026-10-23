"""Tests for the identity-match spike's VMS cut: box buffer and
de-duplication."""

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "spike_identity.py"
T0 = datetime(2024, 5, 27, 12, 0, tzinfo=timezone.utc)


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
