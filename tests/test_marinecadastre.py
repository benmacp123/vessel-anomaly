"""Tests for the US blocks, the vessel split, the extract writer helpers
and the reception grid."""

import hashlib
import hmac
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from vessels.io.blocks import BLOCKS, vessel_group, vessel_id
from vessels.io.marinecadastre import (
    EXTRACT_SCHEMA,
    RECEPTION_SCHEMA,
    fishing_extract,
    pseudonymise,
    read_extract,
    read_reception,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_us_extract.py"
SALT = b"test-salt"
T0 = datetime(2024, 5, 6, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(name="build")
def fixture_build():
    """Load the build script as a module, since ``scripts/`` is not a package.

    Returns:
        module: The loaded build script.
    """
    spec = importlib.util.spec_from_file_location("build_us_extract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_rows(mmsis, types, timestamps=None):
    """A raw frame shaped like ``load_raw_day`` output.

    Args:
        mmsis: MMSI per row.
        types: Vessel type per row.
        timestamps: Time per row (all ``T0`` if omitted).

    Returns:
        pl.DataFrame: One row per MMSI at 70W 42N.
    """
    n = len(mmsis)
    return pl.DataFrame(
        {
            "mmsi": mmsis,
            "timestamp": timestamps or [T0] * n,
            "sog": [3.0] * n,
            "cog": [90.0] * n,
            "heading": [90] * n,
            "status": [7] * n,
            "vessel_type": types,
            "length": [15.0] * n,
            "transceiver": ["B"] * n,
            "lon": [-70.0] * n,
            "lat": [42.0] * n,
        },
        schema_overrides={"mmsi": pl.Int64},
    )


def test_blocks_are_disjoint_weeks():
    """Each block has 28 days, the blocks share none, and neither holds a spike day."""
    tuning, heldout = set(BLOCKS["tuning"]), set(BLOCKS["heldout"])
    assert len(tuning) == len(heldout) == 28
    assert not tuning & heldout
    assert all(day.day != 15 or day.month in (5, 9) for day in tuning | heldout)


def test_vessel_id_is_salted_hmac():
    """The ID is the HMAC-SHA256 prefix and changes with the salt."""
    expected = hmac.new(SALT, b"367000001", hashlib.sha256).hexdigest()[:16]
    assert vessel_id(367000001, SALT) == expected
    assert vessel_id(367000001, b"other") != expected


def test_vessel_group_by_first_digit():
    """Even first hex digits go to A, odd ones to B."""
    assert (vessel_group("0f"), vessel_group("a1"), vessel_group("b0")) == (
        "A",
        "A",
        "B",
    )


def test_pseudonymise_drops_mmsi():
    """No MMSI survives, and each MMSI maps to one ID and group."""
    frame = pseudonymise(raw_rows([1, 1, 2], [30, 30, 70]), SALT)
    assert "mmsi" not in frame.columns
    assert frame.columns[0] == "vessel_id"
    assert frame.select("vessel_id", "group").n_unique() == 2


def test_fishing_extract_keeps_group_and_types():
    """Only fishing types of the block's group stay, in the extract schema."""
    mmsis = list(range(367000000, 367000040))
    frame = pseudonymise(raw_rows(mmsis, [30, 70] * 20), SALT)
    extract = fishing_extract(frame, "A")
    assert dict(extract.schema) == EXTRACT_SCHEMA
    assert extract["vessel_type"].to_list() == [30] * extract.height
    assert {vessel_group(v) for v in extract["vessel_id"]} == {"A"}
    assert 0 < extract.height < 20


def test_fishing_extract_order_is_total():
    """Shuffled input gives an identical extract."""
    mmsis = [367000001] * 3 + [367000002] * 3
    times = [T0] * 6
    frame = pseudonymise(raw_rows(mmsis, [30] * 6, times), SALT).with_columns(
        sog=pl.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=pl.Float64)
    )
    group = vessel_group(frame["vessel_id"][0])
    first = fishing_extract(frame, group)
    second = fishing_extract(frame.sample(fraction=1.0, shuffle=True, seed=3), group)
    assert first.equals(second)


def test_read_extract_checks_schema(tmp_path):
    """A file in another schema is refused."""
    good = tmp_path / "good.parquet"
    pl.DataFrame(schema=EXTRACT_SCHEMA).write_parquet(good)
    assert read_extract([good]).collect().height == 0
    bad = tmp_path / "bad.parquet"
    pl.DataFrame({"mmsi": [1]}).write_parquet(bad)
    with pytest.raises(ValueError, match="unexpected extract schema"):
        read_extract([bad])


def test_reception_grid_counts_positions_and_vessels(build):
    """Positions and distinct vessels are counted per cell and class."""
    frame = pseudonymise(raw_rows([1, 1, 2], [30, 30, 70]), SALT)
    grid = build.reception_grid(frame)
    assert grid.select("positions", "vessels").rows() == [(3, 2)]


def test_reception_grid_counts_moving_and_long_intervals(build):
    """Intervals count from moving starts only; over 10 minutes is long."""
    minutes = [0, 1, 1, 12, 13, 30]
    frame = pseudonymise(
        raw_rows([1] * 6, [70] * 6, [T0 + timedelta(minutes=m) for m in minutes]),
        SALT,
    ).with_columns(sog=pl.Series([3.0, 3.0, 3.0, 3.0, 0.5, 3.0]))
    grid = build.reception_grid(frame)
    # 0-1 counts; 1-1 is a repeated time; 1-12 is long; 12-13 counts; the
    # slow start at 13 does not count; 30 has no next position that day.
    assert grid.select("moving_intervals", "long_intervals").rows() == [(3, 1)]


def test_reception_grid_writes_the_schema_the_reader_expects(build, tmp_path):
    """The builder's grid passes the reader's schema check; another schema fails."""
    frame = pseudonymise(raw_rows([1, 2], [30, 70]), SALT)
    good = tmp_path / "good.parquet"
    build.reception_grid(frame).write_parquet(good)
    assert dict(read_reception([good]).collect_schema()) == RECEPTION_SCHEMA
    bad = tmp_path / "bad.parquet"
    pl.DataFrame({"h3_cell": ["x"]}).write_parquet(bad)
    with pytest.raises(ValueError, match="unexpected reception schema"):
        read_reception([bad])
