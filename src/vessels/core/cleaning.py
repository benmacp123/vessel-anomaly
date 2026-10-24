"""Clean AIS positions: bad values, duplicates and positions well inland.

Rules follow GFW's pipe-segment (``transform/invalid_values.py``,
Apache-2.0; literature-sourced (GFW)): an out-of-range position drops the
row, but an out-of-range speed, course or heading only becomes null, since
the position itself is still usable. On top of that:

- MMSI validity (length, placeholders, MID outside 201-775 per USCG
  NAVCEN) is an expression for the extract builder, because committed
  extracts carry only hashed IDs;
- rows repeating a vessel, time and position are dropped (the same
  message heard twice);
- optionally, positions more than a buffer inside land are dropped. The
  US pipeline does not use this: on the tuning block 35% of positions lie
  inside Natural Earth 10m land (95% of them moored), and those more than
  2 km inland are mostly real upriver harbours and Cape Cod Canal
  transits, so no buffer separates errors from harbours. Dropping them
  would turn port stays into AIS gaps; off-track jumps are left to the
  segmenter instead (observed, 2026-10-24).

Every step is order-independent: output is sorted on every column, so the
same input rows in any order, with any repeats, give the same frame.
"""

from dataclasses import dataclass, field

import polars as pl
import shapely
from shapely.geometry.base import BaseGeometry

from vessels.core.geo import LocalProjection

# Speeds GFW treats as invalid: 102.3 kn is "not available" in AIS.
MAX_SOG = 102.3
# Literature-sourced (GFW): receivers report spurious speeds at these
# values. The tolerance is provisional: SOG is reported to 0.1 kn.
NOISE_SOG = (51.2, 63.0)
NOISE_TOLERANCE = 0.05
PLACEHOLDER_MMSIS = (123456789, 1193046)
# USCG NAVCEN: ship MIDs run from 201 to 775.
MID_RANGE = (201, 775)
# Rows dropped, in the order the rules apply.
DROP_RULES = ("bad_position", "duplicate", "inland")
# Values set to null; the row is kept.
NULL_RULES = ("sog_invalid", "sog_noise", "cog_invalid", "heading_invalid")
DEDUP_KEY = ("vessel_id", "timestamp", "lon", "lat")


@dataclass
class CleanResult:
    """Cleaned positions and what each rule did.

    Attributes:
        frame: Cleaned rows, sorted by vessel, time and then every column.
        dropped: Rows removed per rule in ``DROP_RULES``.
        nulled: Values set to null per rule in ``NULL_RULES``.
    """

    frame: pl.DataFrame
    dropped: dict[str, int] = field(default_factory=dict)
    nulled: dict[str, int] = field(default_factory=dict)


def mmsi_rule(mmsi: pl.Expr) -> pl.Expr:
    """Name the first validity rule an MMSI breaks, or null if valid.

    A placeholder is named before its length so that 1193046 (hex 123456)
    is counted as a placeholder.

    Args:
        mmsi: Integer MMSI expression.

    Returns:
        pl.Expr: ``"placeholder"``, ``"length"``, ``"mid"`` or null.
    """
    mid = mmsi // 1_000_000
    return (
        pl.when(mmsi.is_in(PLACEHOLDER_MMSIS))
        .then(pl.lit("placeholder"))
        .when((mmsi < 100_000_000) | (mmsi > 999_999_999))
        .then(pl.lit("length"))
        .when((mid < MID_RANGE[0]) | (mid > MID_RANGE[1]))
        .then(pl.lit("mid"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )


def inland_area(
    land: BaseGeometry, buffer_km: float, projection: LocalProjection
) -> BaseGeometry:
    """Land shrunk by a buffer, in projected kilometres.

    Args:
        land: Land polygons in longitude and latitude.
        buffer_km: How far inside the coastline a position must be to drop.
        projection: Projection the points will be tested in.

    Returns:
        BaseGeometry: Prepared land interior in kilometres.
    """
    area = projection.geometry(land).buffer(-buffer_km)
    shapely.prepare(area)
    return area


def _sort_all(frame: pl.DataFrame) -> pl.DataFrame:
    """Sort on vessel, time and then every other column, nulls last.

    Args:
        frame: Positions.

    Returns:
        pl.DataFrame: The rows in a total order independent of input order.
    """
    rest = [c for c in frame.columns if c not in ("vessel_id", "timestamp")]
    return frame.sort(["vessel_id", "timestamp", *rest], nulls_last=True)


def _null_invalid(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Set invalid speed, course and heading values to null.

    Args:
        frame: Positions with ``sog``, ``cog`` and ``heading``.

    Returns:
        tuple[pl.DataFrame, dict[str, int]]: The frame and values nulled
        per rule.
    """
    sog = pl.col("sog")
    noise = pl.any_horizontal((sog - s).abs() <= NOISE_TOLERANCE for s in NOISE_SOG)
    flags = {
        "sog_invalid": (sog < 0) | (sog >= MAX_SOG),
        "sog_noise": noise & (sog < MAX_SOG),
        "cog_invalid": (pl.col("cog") < 0) | (pl.col("cog") >= 360),
        "heading_invalid": (pl.col("heading") < 0) | (pl.col("heading") >= 360),
    }
    counts = frame.select(
        **{name: flag.fill_null(False).sum() for name, flag in flags.items()}
    ).row(0, named=True)
    frame = frame.with_columns(
        sog=pl.when(flags["sog_invalid"] | noise).then(None).otherwise(sog),
        cog=pl.when(flags["cog_invalid"]).then(None).otherwise(pl.col("cog")),
        heading=pl.when(flags["heading_invalid"])
        .then(None)
        .otherwise(pl.col("heading")),
    )
    return frame, {name: int(counts[name]) for name in NULL_RULES}


def clean(
    frame: pl.DataFrame,
    inland: BaseGeometry | None = None,
    projection: LocalProjection | None = None,
) -> CleanResult:
    """Apply every cleaning rule to positions in the internal schema.

    Args:
        frame: Positions with ``vessel_id``, ``timestamp``, ``lon``,
            ``lat``, ``sog``, ``cog`` and ``heading``.
        inland: ``inland_area`` output; None skips the land mask.
        projection: The projection ``inland`` was built in.

    Returns:
        CleanResult: Cleaned rows and counts per rule.

    Raises:
        ValueError: If ``inland`` is given without ``projection``.
    """
    if inland is not None and projection is None:
        raise ValueError("inland needs the projection it was built in")
    dropped = {}
    lon, lat = pl.col("lon"), pl.col("lat")
    good = (
        lon.is_between(-180, 180) & lat.is_between(-90, 90) & ~((lon == 0) & (lat == 0))
    ).fill_null(False)
    kept = frame.filter(good)
    dropped["bad_position"] = frame.height - kept.height

    kept, nulled = _null_invalid(kept)
    kept = _sort_all(kept)

    before = kept.height
    kept = kept.unique(subset=list(DEDUP_KEY), keep="first", maintain_order=True)
    dropped["duplicate"] = before - kept.height

    before = kept.height
    if inland is not None:
        x, y = projection.forward(kept["lon"].to_numpy(), kept["lat"].to_numpy())
        kept = kept.filter(~pl.Series(shapely.contains_xy(inland, x, y)))
    dropped["inland"] = before - kept.height
    return CleanResult(kept, dropped, nulled)
