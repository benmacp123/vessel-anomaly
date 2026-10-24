"""Receiver coverage per H3 cell and AIS class, from daily reception counts.

A cell counts as covered for a class when moving vessels there are heard
without dropouts: the share of their position-to-position intervals over
10 minutes (counted in the extract build) is low, and there are enough
intervals to show it. One rule weighs both: the 95% Wilson upper bound
of that share must not exceed ``MAX_LONG_SHARE``. With no long intervals
at all, that needs about 35 moving intervals in the cell.

Three kinds of cell are not covered: cells where reception drops out,
cells with too little moving traffic to tell (no evidence either way),
and cells cut by the study box, where a vessel leaving the box looks like
a dropout. A gap detector should treat all three alike, so the distance
to the nearest uncovered cell is measured to any of them, or to a cell
outside the box.

Thresholds are provisional, set on the tuning block (2026-10-24). The
measure can also be raised by vessels switching AIS off while moving, so
it bounds reception quality from below.
"""

import h3
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from vessels.core.geo import LocalProjection

# Provisional: chosen on the tuning block, where 99% of well-sampled
# Class A cells and 95% of Class B cells have a long share under 0.1.
MAX_LONG_SHARE = 0.1
# Two-sided 95% normal quantile for the Wilson interval.
Z95 = 1.959964
# Degrees added around the box when listing cells outside it, so the box
# edge always has an uncovered cell beyond it.
OUTSIDE_MARGIN = 0.1


def wilson_upper(successes: pl.Expr, trials: pl.Expr, z: float = Z95) -> pl.Expr:
    """Upper bound of the Wilson score interval for a proportion.

    Args:
        successes: Count of events.
        trials: Count of trials; zero gives an upper bound of 1.
        z: Normal quantile for the confidence level.

    Returns:
        pl.Expr: The upper bound, between 0 and 1.
    """
    n = trials.cast(pl.Float64)
    p = successes.cast(pl.Float64) / n
    centre = p + z**2 / (2 * n)
    spread = z * (p * (1 - p) / n + z**2 / (4 * n**2)).sqrt()
    upper = (centre + spread) / (1 + z**2 / n)
    return pl.when(n > 0).then(upper.clip(0.0, 1.0)).otherwise(1.0)


def summarise(grid: pl.DataFrame) -> pl.DataFrame:
    """Total daily reception rows per cell and class.

    Args:
        grid: Daily reception rows: ``h3_cell``, ``transceiver``,
            ``positions``, ``vessels``, ``moving_intervals`` and
            ``long_intervals``, one row per cell, class and day.

    Returns:
        pl.DataFrame: Per cell and class, ``days`` with any reception,
        ``positions``, ``vessel_days``, ``moving_intervals``,
        ``long_intervals``, ``long_share`` (null without moving
        intervals) and ``long_share_upper``, sorted by cell and class.
    """
    return (
        grid.group_by("h3_cell", "transceiver")
        .agg(
            days=pl.len().cast(pl.Int64),
            positions=pl.col("positions").sum(),
            vessel_days=pl.col("vessels").sum(),
            moving_intervals=pl.col("moving_intervals").sum(),
            long_intervals=pl.col("long_intervals").sum(),
        )
        .with_columns(
            long_share=pl.when(pl.col("moving_intervals") > 0).then(
                pl.col("long_intervals") / pl.col("moving_intervals")
            ),
            long_share_upper=wilson_upper(
                pl.col("long_intervals"), pl.col("moving_intervals")
            ),
        )
        .sort("h3_cell", "transceiver", nulls_last=True)
    )


def cut_by_box(cell: str, box: tuple[float, float, float, float]) -> bool:
    """Whether any corner of a cell lies outside the box.

    Args:
        cell: H3 cell.
        box: ``(min_lon, min_lat, max_lon, max_lat)``.

    Returns:
        bool: True if the cell is not wholly inside the box.
    """
    min_lon, min_lat, max_lon, max_lat = box
    return any(
        not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat)
        for lat, lon in h3.cell_to_boundary(cell)
    )


def cells_around(box: tuple[float, float, float, float], resolution: int) -> set:
    """Every cell whose centre lies in the box widened by ``OUTSIDE_MARGIN``.

    Args:
        box: ``(min_lon, min_lat, max_lon, max_lat)``.
        resolution: H3 resolution.

    Returns:
        set: H3 cells.
    """
    min_lon, min_lat, max_lon, max_lat = (
        box[0] - OUTSIDE_MARGIN,
        box[1] - OUTSIDE_MARGIN,
        box[2] + OUTSIDE_MARGIN,
        box[3] + OUTSIDE_MARGIN,
    )
    outline = h3.LatLngPoly(
        [(min_lat, min_lon), (min_lat, max_lon), (max_lat, max_lon), (max_lat, min_lon)]
    )
    return set(h3.polygon_to_cells(outline, resolution))


def _centres(cells: list[str], projection: LocalProjection) -> np.ndarray:
    """Project cell centres to kilometres.

    Args:
        cells: H3 cells.
        projection: Local projection for the study box.

    Returns:
        np.ndarray: One ``(x, y)`` row per cell.
    """
    lat, lon = np.array([h3.cell_to_latlng(c) for c in cells]).T
    return np.column_stack(projection.forward(lon, lat))


def coverage(
    summary: pl.DataFrame,
    box: tuple[float, float, float, float],
    max_long_share: float = MAX_LONG_SHARE,
) -> pl.DataFrame:
    """Mark covered cells and their distance to the nearest uncovered one.

    Args:
        summary: ``summarise`` output.
        box: Study box ``(min_lon, min_lat, max_lon, max_lat)``.
        max_long_share: Largest allowed Wilson upper bound of the long
            interval share.

    Returns:
        pl.DataFrame: ``summary`` plus ``cut_by_box``, ``covered``,
        ``status`` (``covered``, ``dropouts``, ``too few intervals`` or
        ``box edge``) and ``edge_km`` (centre-to-centre kilometres from
        a covered cell to the nearest uncovered cell of the same class,
        in or around the box; null when not covered).
    """
    cut = [cut_by_box(c, box) for c in summary["h3_cell"]]
    marked = (
        summary.with_columns(cut_by_box=pl.Series(cut, dtype=pl.Boolean))
        .with_columns(
            covered=~pl.col("cut_by_box")
            & (pl.col("long_share_upper") <= max_long_share)
        )
        .with_columns(
            # Why a cell is uncovered: if it had enough intervals to pass
            # with no long ones at all, its long intervals failed it
            # (dropouts); otherwise there were too few intervals to tell.
            status=pl.when(pl.col("cut_by_box"))
            .then(pl.lit("box edge"))
            .when(pl.col("covered"))
            .then(pl.lit("covered"))
            .when(wilson_upper(pl.lit(0), pl.col("moving_intervals")) <= max_long_share)
            .then(pl.lit("dropouts"))
            .otherwise(pl.lit("too few intervals"))
        )
    )
    if not summary.height:
        return marked.with_columns(edge_km=pl.lit(None, dtype=pl.Float64))
    projection = LocalProjection((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    around = cells_around(box, h3.get_resolution(summary["h3_cell"][0]))
    parts = []
    for (cls,), rows in marked.group_by("transceiver", maintain_order=True):
        covered = rows.filter(pl.col("covered"))["h3_cell"].to_list()
        uncovered = sorted(around - set(covered))
        edge = np.full(len(covered), np.nan)
        if covered and uncovered:
            edge, _ = cKDTree(_centres(uncovered, projection)).query(
                _centres(covered, projection)
            )
        parts.append(
            pl.DataFrame(
                {
                    "h3_cell": covered,
                    "transceiver": [cls] * len(covered),
                    "edge_km": edge,
                },
                schema={
                    "h3_cell": pl.String,
                    "transceiver": pl.String,
                    "edge_km": pl.Float64,
                },
            )
        )
    return (
        marked.join(
            pl.concat(parts),
            on=["h3_cell", "transceiver"],
            how="left",
            nulls_equal=True,
        )
        .with_columns(pl.col("edge_km").fill_nan(None))
        .sort("h3_cell", "transceiver", nulls_last=True)
    )
