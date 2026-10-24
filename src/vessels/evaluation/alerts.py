"""Real-data alert rates with Poisson intervals (PLAN.md, evaluation item 3).

Real alerts have no truth, so they are reported as rates, never as
precision: gaps and loitering per 1,000 vessel-days, closure events per
1,000 fishing-vessel-hours near each closure, every detector per day for
the region.

Gaps are reported three ways, so the coverage mask's effect can be seen
(PLAN.md, evaluation item 7):

- ``rule``: the unexplained-gap rule as built;
- ``no_mask``: the same rule without the coverage and cells-heard checks;
- ``gfw_50nm``: GFW's published rule, approximated: at least 12 hours,
  both ends over 50 nm from shore, at least 14 positions in the prior
  12 hours. Its reception condition (over 10 positions a day in the 1°
  cell) is not applied: it targets satellite data.

Each variant is stratified by where the gap starts: ``uncovered`` cells,
or covered cells banded by distance to the coverage edge.
"""

import numpy as np
import polars as pl
import shapely
from shapely.geometry.base import BaseGeometry

from vessels.core.distance import ShoreDistance
from vessels.core.geo import KM_PER_NM, LocalProjection
from vessels.evaluation.metrics import poisson_rate

# Provisional bands of distance to the coverage edge (km), fixed before
# any held-out read.
EDGE_BANDS_KM = (10.0, 30.0)
# Distance around a closure that counts as "near" for exposure.
NEAR_KM = 10.0
# Longest step counted as time spent somewhere.
MAX_STEP_HOURS = 1.0
# GFW's offshore condition for intentional disabling (Welch et al. 2022).
GFW_OFFSHORE_NM = 50.0
GFW_MIN_HOURS = 12.0
ALERT_SCHEMA = {
    "detector": pl.String,
    "variant": pl.String,
    "transceiver": pl.String,
    "stratum": pl.String,
    "events": pl.Int64,
    "vessels": pl.Int64,
    "exposure": pl.Float64,
    "unit": pl.String,
    "rate": pl.Float64,
    "low": pl.Float64,
    "high": pl.Float64,
}


def vessel_days(positions: pl.DataFrame) -> dict[str, int]:
    """Distinct vessel and UTC day pairs, per AIS class and in all.

    Args:
        positions: Positions with ``vessel_id``, ``timestamp`` and
            ``transceiver``.

    Returns:
        dict[str, int]: Count by class, plus ``all``.
    """
    days = positions.select(
        "vessel_id", "transceiver", day=pl.col("timestamp").dt.date()
    ).unique()
    counts = dict(days.group_by("transceiver").len().iter_rows())
    counts["all"] = days.select("vessel_id", "day").unique().height
    return counts


def _rate_row(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    detector: str,
    variant: str,
    transceiver: str,
    stratum: str,
    events: pl.DataFrame,
    exposure: float,
    unit: str,
    per: float,
) -> dict:
    """One rate row.

    Args:
        detector: Detector name.
        variant: Rule variant.
        transceiver: AIS class or ``all``.
        stratum: Stratum label.
        events: The events counted.
        exposure: Exposure in ``unit``.
        unit: Exposure unit.
        per: Rate scale.

    Returns:
        dict: A row in ``ALERT_SCHEMA``.
    """
    rate, low, high = (
        poisson_rate(events.height, exposure, per) if exposure > 0 else (None,) * 3
    )
    return {
        "detector": detector,
        "variant": variant,
        "transceiver": transceiver,
        "stratum": stratum,
        "events": events.height,
        "vessels": events["vessel_id"].n_unique() if events.height else 0,
        "exposure": round(exposure, 1),
        "unit": f"per {per:g} {unit}",
        "rate": None if rate is None else round(rate, 3),
        "low": None if low is None else round(low, 3),
        "high": None if high is None else round(high, 3),
    }


def gap_variants(gaps: pl.DataFrame, shore: ShoreDistance) -> dict[str, pl.DataFrame]:
    """The gaps each variant would alert on.

    Args:
        gaps: ``gap_events`` output.
        shore: Distance to shore.

    Returns:
        dict[str, pl.DataFrame]: Gaps by variant name.
    """
    start_nm = shore.km(gaps["start_lon"], gaps["start_lat"]) / KM_PER_NM
    end_nm = shore.km(gaps["end_lon"], gaps["end_lat"]) / KM_PER_NM
    offshore = pl.Series((start_nm > GFW_OFFSHORE_NM) & (end_nm > GFW_OFFSHORE_NM))
    return {
        "rule": gaps.filter("unexplained"),
        "no_mask": gaps.filter(
            pl.col("away_from_port")
            & pl.col("away_from_mooring")
            & pl.col("dense_before")
        ),
        "gfw_50nm": gaps.filter(
            offshore & (pl.col("hours") >= GFW_MIN_HOURS) & pl.col("dense_before")
        ),
    }


def edge_strata(gaps: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """Label each gap by its start cell's coverage and edge distance.

    Args:
        gaps: Gap rows with ``start_cell`` and ``transceiver``.
        grid: Coverage grid with ``covered`` and ``edge_km``.

    Returns:
        pl.DataFrame: ``gaps`` plus ``stratum``.
    """
    low, high = EDGE_BANDS_KM
    return (
        gaps.join(
            grid.select("h3_cell", "transceiver", "covered", "edge_km").rename(
                {"covered": "_covered"}
            ),
            left_on=["start_cell", "transceiver"],
            right_on=["h3_cell", "transceiver"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            stratum=pl.when(pl.col("_covered").fill_null(False).not_())
            .then(pl.lit("uncovered"))
            .when(pl.col("edge_km") < low)
            .then(pl.lit(f"edge under {low:g} km"))
            .when(pl.col("edge_km") < high)
            .then(pl.lit(f"edge {low:g}-{high:g} km"))
            .otherwise(pl.lit(f"edge {high:g} km+"))
        )
        .drop("_covered", "edge_km")
    )


def gap_rates(
    gaps: pl.DataFrame, grid: pl.DataFrame, shore: ShoreDistance, days: dict[str, int]
) -> list[dict]:
    """Gap alert rates per variant, class and stratum.

    Args:
        gaps: ``gap_events`` output.
        grid: Coverage grid.
        shore: Distance to shore.
        days: ``vessel_days`` output.

    Returns:
        list[dict]: Rows in ``ALERT_SCHEMA``.
    """
    rows = []
    for variant, found in gap_variants(gaps, shore).items():
        found = edge_strata(found, grid)
        for transceiver in ("A", "B"):
            mine = found.filter(pl.col("transceiver") == transceiver)
            exposure = days.get(transceiver, 0)
            rows.append(
                _rate_row(
                    "gap",
                    variant,
                    transceiver,
                    "all",
                    mine,
                    exposure,
                    "vessel-days",
                    1000,
                )
            )
            for (stratum,), part in sorted(mine.group_by("stratum")):
                rows.append(
                    _rate_row(
                        "gap",
                        variant,
                        transceiver,
                        stratum,
                        part,
                        exposure,
                        "vessel-days",
                        1000,
                    )
                )
    return rows


def hours_near(
    positions: pl.DataFrame, area: BaseGeometry, projection: LocalProjection
) -> float:
    """Fishing-vessel hours spent within ``NEAR_KM`` of an area.

    Each position stands for the time to the vessel's next one, capped at
    ``MAX_STEP_HOURS``.

    Args:
        positions: Positions sorted by vessel and time.
        area: Polygon in longitude and latitude.
        projection: Projection for the buffer.

    Returns:
        float: Hours.
    """
    near = projection.geometry(area).buffer(NEAR_KM)
    x, y = projection.forward(positions["lon"].to_numpy(), positions["lat"].to_numpy())
    inside = shapely.contains_xy(near, x, y)
    step = (
        positions.select(
            (pl.col("timestamp").shift(-1).over("vessel_id") - pl.col("timestamp"))
            .dt.total_microseconds()
            .truediv(3.6e9)
            .fill_null(0.0)
            .clip(0.0, MAX_STEP_HOURS)
        )
        .to_series()
        .to_numpy()
    )
    return float(np.sum(step[inside]))


def closure_rates(
    events: pl.DataFrame,
    positions: pl.DataFrame,
    areas: dict[str, BaseGeometry],
    projection: LocalProjection,
) -> list[dict]:
    """Apparent closure fishing per 1,000 fishing-vessel-hours nearby.

    Args:
        events: ``closure_events`` output.
        positions: The block's positions.
        areas: Closure ID to polygon (every closure with events or scored).
        projection: Projection for the buffer.

    Returns:
        list[dict]: Rows in ``ALERT_SCHEMA``, one per closure and reason.
    """
    ordered = positions.sort("vessel_id", "timestamp")
    rows = []
    for closure in sorted(areas):
        exposure = hours_near(ordered, areas[closure], projection)
        for reason in ("closure", "suspended_exemption"):
            found = events.filter(
                (pl.col("closure_id") == closure) & (pl.col("reason") == reason)
            )
            if reason != "closure" and found.is_empty():
                continue
            rows.append(
                _rate_row(
                    "closure_fishing",
                    reason,
                    "all",
                    closure,
                    found,
                    exposure,
                    f"vessel-hours within {NEAR_KM:g} km",
                    1000,
                )
            )
    return rows


def per_vessel_day_rates(
    detector: str, variant: str, events: pl.DataFrame, days: dict[str, int]
) -> list[dict]:
    """Events per 1,000 vessel-days, per class and in all.

    Args:
        detector: Detector name.
        variant: Rule variant.
        events: Events with ``vessel_id`` and ``transceiver``.
        days: ``vessel_days`` output.

    Returns:
        list[dict]: Rows in ``ALERT_SCHEMA``.
    """
    rows = [
        _rate_row(
            detector,
            variant,
            transceiver,
            "all",
            events.filter(pl.col("transceiver") == transceiver),
            days.get(transceiver, 0),
            "vessel-days",
            1000,
        )
        for transceiver in ("A", "B")
    ]
    rows.append(
        _rate_row(
            detector, variant, "all", "all", events, days["all"], "vessel-days", 1000
        )
    )
    return rows


def region_rate(detector: str, variant: str, events: pl.DataFrame, days: int) -> dict:
    """Events per day for the whole region.

    Args:
        detector: Detector name.
        variant: Rule variant.
        events: Events.
        days: Days in the block.

    Returns:
        dict: A row in ``ALERT_SCHEMA``.
    """
    return _rate_row(detector, variant, "all", "region", events, days, "days", 1)
