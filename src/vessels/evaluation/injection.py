"""Run the planted-event scenarios through the detectors and score them.

Each kind of case (gap, closure, encounter, loitering) is planted in its own
copy of the block, so cases of different kinds cannot interact, and the
batch detectors run once per copy. A planted case is hit when a detection
with the same keys overlaps it (``metrics.match``): an unexplained gap of
the vessel, a closure event of the vessel in that closure (reason
``closure``), an encounter of the pair, or a loitering event of the vessel.

Recall is stratified by AIS class and by reception density: the tuning
grid's mean positions per day in the case's starting cell for its class,
in fixed bands (``DENSITY_BANDS``).
"""

from dataclasses import dataclass, replace

import h3
import numpy as np
import polars as pl
import shapely
from shapely.geometry.base import BaseGeometry

from vessels.core.closure_fishing import DETECTOR as CLOSURE
from vessels.core.encounters import DETECTOR as ENCOUNTER
from vessels.core.gaps import DETECTOR as GAP
from vessels.core.geo import KM_PER_NM
from vessels.core.geofence import Rule
from vessels.evaluation.metrics import match, wilson
from vessels.sim.inject import InjectionRules, plant_closures, plant_encounters
from vessels.sim.inject import plant_gaps, plant_loitering
from vessels.stream.micro import Inputs, run_batch

# Provisional bands of mean positions per cell-day (tuning grid, the case's
# class), fixed before any held-out read: sparse, moderate, dense.
DENSITY_BANDS = (50.0, 500.0)
# Real gaps whose lengths positives are drawn from: over the rule's 6 hours
# and at most two days, so planted windows still fit inside a block.
LENGTH_RANGE_HOURS = (6.0, 48.0)
# Tolerance on interval ends when matching (the resampling grid).
SLACK_MINUTES = 10.0
# Stream replay for detection delay: the Phase 3 replay's arrival pattern,
# hourly batches, a fixed seed.
ARRIVAL = "backhaul"
EVERY_HOURS = 1.0
ARRIVAL_SEED = 0


@dataclass(frozen=True)
class Scenario:
    """One planted copy of a block and what the detectors found in it.

    Attributes:
        kind: Case kind (a detector name).
        positions: Positions with the planted cases.
        truth: Truth table with ``hit`` and ``found_start``.
        found: The detector's events on the planted copy.
        inputs: Detector inputs for the copy (a gap copy's reception has
            the planted silences taken out).
    """

    kind: str
    positions: pl.DataFrame
    truth: pl.DataFrame
    found: pl.DataFrame
    inputs: Inputs


def annotate(positions: pl.DataFrame, inputs: Inputs) -> pl.DataFrame:
    """Add what the planters need to each position.

    Args:
        positions: Positions in the extract schema.
        inputs: Detector inputs (grid, anchorages, shore).

    Returns:
        pl.DataFrame: ``positions`` plus ``h3_cell``, ``covered``,
        ``density`` (mean positions per day in the cell for the class, from
        the tuning grid), ``port_km`` and ``shore_nm``.
    """
    resolution = h3.get_resolution(inputs.grid["h3_cell"][0])
    lon, lat = positions["lon"].to_numpy(), positions["lat"].to_numpy()
    cells = pl.Series(
        [h3.latlng_to_cell(y, x, resolution) for x, y in zip(lon, lat)],
        dtype=pl.String,
    )
    grid = inputs.grid.select(
        "h3_cell",
        "transceiver",
        "covered",
        density=pl.col("positions") / pl.col("days"),
    )
    return (
        positions.with_columns(
            h3_cell=cells,
            port_km=pl.Series(inputs.ports.km(lon, lat)),
            shore_nm=pl.Series(inputs.shore.km(lon, lat) / KM_PER_NM),
        )
        .join(grid, on=["h3_cell", "transceiver"], how="left", maintain_order="left")
        .with_columns(
            pl.col("covered").fill_null(False), pl.col("density").fill_null(0.0)
        )
    )


def closure_areas(
    rules: list[Rule], scored: tuple[str, ...]
) -> dict[str, BaseGeometry]:
    """Scored closures with every exemption area cut out of them.

    Args:
        rules: The geofence table.
        scored: Closure IDs to keep.

    Returns:
        dict[str, BaseGeometry]: Closure ID to polygon.
    """
    areas = {}
    for rule in rules:
        if rule.area_id not in scored:
            continue
        cut = [r.geometry for r in rules if r.parent == rule.area_id]
        areas[rule.area_id] = (
            shapely.difference(rule.geometry, shapely.union_all(cut))
            if cut
            else rule.geometry
        )
    return areas


def gap_lengths(gaps: pl.DataFrame) -> np.ndarray:
    """Real gap lengths to draw planted positives from.

    Args:
        gaps: ``gap_events`` output of a block (the tuning block).

    Returns:
        np.ndarray: Hours of every gap within ``LENGTH_RANGE_HOURS``.
    """
    low, high = LENGTH_RANGE_HOURS
    return gaps.filter(pl.col("hours").is_between(low, high))["hours"].to_numpy()


def _found(kind: str, events: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Detections that count as hits for a kind, with the keys to match on.

    Args:
        kind: Detector name.
        events: That detector's events.

    Returns:
        tuple[pl.DataFrame, list[str]]: Detections and keys.
    """
    if kind == GAP:
        return events.filter("unexplained"), ["vessel_id"]
    if kind == CLOSURE:
        return events.filter(pl.col("reason") == "closure"), ["vessel_id", "closure_id"]
    if kind == ENCOUNTER:
        pair = events.with_columns(
            pair=pl.concat_str(
                pl.min_horizontal("vessel_id", "partner_id"),
                pl.max_horizontal("vessel_id", "partner_id"),
                separator="|",
            )
        )
        return pair, ["pair"]
    return events, ["vessel_id"]


def _keyed_truth(kind: str, truth: pl.DataFrame) -> pl.DataFrame:
    """Add the encounter pair key to a truth table.

    Args:
        kind: Detector name.
        truth: Truth table.

    Returns:
        pl.DataFrame: ``truth``, with ``pair`` for encounters.
    """
    if kind != ENCOUNTER:
        return truth
    return truth.with_columns(
        pair=pl.concat_str(
            pl.min_horizontal("vessel_id", "partner_id"),
            pl.max_horizontal("vessel_id", "partner_id"),
            separator="|",
        )
    )


def silenced(
    reception: pl.DataFrame, annotated: pl.DataFrame, planted: pl.DataFrame
) -> pl.DataFrame:
    """Cells and days still heard once planted silences are taken out.

    The reception grid counts every position a receiver heard, including
    the ones a planted gap deletes; left in, they would make a planted gap
    look heard where a real one would not be.

    Args:
        reception: ``inputs.reception_days`` for the block.
        annotated: ``annotate`` output (its ``h3_cell`` is the reception
            grid's resolution).
        planted: Positions after planting.

    Returns:
        pl.DataFrame: ``h3_cell`` and ``day`` with traffic left.
    """
    deleted = (
        annotated.join(
            planted.select("vessel_id", "timestamp", "lon", "lat"),
            on=["vessel_id", "timestamp", "lon", "lat"],
            how="anti",
        )
        .group_by("h3_cell", day=pl.col("timestamp").dt.date())
        .agg(deleted=pl.len())
    )
    return (
        reception.join(deleted, on=["h3_cell", "day"], how="left")
        .filter(pl.col("positions") > pl.col("deleted").fill_null(0))
        .select("h3_cell", "day")
    )


def plant(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    kind: str,
    annotated: pl.DataFrame,
    inputs: Inputs,
    lengths: np.ndarray,
    scored: tuple[str, ...],
    rules: InjectionRules = InjectionRules(),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Plant one kind of case.

    Args:
        kind: Detector name.
        annotated: ``annotate`` output.
        inputs: Detector inputs.
        lengths: Real gap lengths for planted gaps.
        scored: Closure IDs to plant runs in.
        rules: Generator settings.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Planted positions and truth.
    """
    projection = inputs.ports.projection
    if kind == GAP:
        return plant_gaps(annotated, lengths, rules)
    if kind == CLOSURE:
        areas = closure_areas(inputs.closures, scored)
        return plant_closures(annotated, areas, projection, rules)
    if kind == ENCOUNTER:
        return plant_encounters(annotated, projection, rules)
    return plant_loitering(annotated, projection, rules)


def run_scenario(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    kind: str,
    annotated: pl.DataFrame,
    inputs: Inputs,
    reception: pl.DataFrame,
    lengths: np.ndarray,
    scored: tuple[str, ...],
    rules: InjectionRules = InjectionRules(),
) -> Scenario:
    """Plant one kind of case, run the batch detectors and match.

    Args:
        kind: Detector name.
        annotated: ``annotate`` output.
        inputs: Detector inputs.
        reception: ``inputs.reception_days`` for the block.
        lengths: Real gap lengths for planted gaps.
        scored: Closure IDs to plant runs in.
        rules: Generator settings.

    Returns:
        Scenario: The planted copy, its truth with hits, and detections.
    """
    positions, truth = plant(kind, annotated, inputs, lengths, scored, rules)
    if kind == GAP:
        inputs = replace(inputs, heard=silenced(reception, annotated, positions))
    events = run_batch(positions, inputs)[kind]
    found, keys = _found(kind, events)
    truth = _keyed_truth(kind, truth)
    matched = match(truth, found, keys, SLACK_MINUTES)
    return Scenario(kind, positions, matched, events, inputs)


def with_density(truth: pl.DataFrame, annotated: pl.DataFrame) -> pl.DataFrame:
    """Attach each case's reception density band.

    Args:
        truth: Truth table with ``lon``, ``lat`` and ``transceiver``.
        annotated: ``annotate`` output, for the grid's resolution.

    Returns:
        pl.DataFrame: ``truth`` plus ``density`` and ``density_band``.
    """
    resolution = h3.get_resolution(annotated["h3_cell"][0])
    cells = pl.Series(
        [
            h3.latlng_to_cell(y, x, resolution)
            for x, y in zip(truth["lon"], truth["lat"])
        ],
        dtype=pl.String,
    )
    density = annotated.select("h3_cell", "transceiver", "density").unique(
        ["h3_cell", "transceiver"]
    )
    low, high = DENSITY_BANDS
    return (
        truth.with_columns(h3_cell=cells)
        .join(density, on=["h3_cell", "transceiver"], how="left", maintain_order="left")
        .with_columns(pl.col("density").fill_null(0.0))
        .with_columns(
            density_band=pl.when(pl.col("density") < low)
            .then(pl.lit(f"under {low:g}"))
            .when(pl.col("density") < high)
            .then(pl.lit(f"{low:g}-{high:g}"))
            .otherwise(pl.lit(f"{high:g}+"))
        )
        .drop("h3_cell")
    )


def recall_table(truth: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Hits per group, with Wilson intervals.

    Args:
        truth: Matched truth (``hit``) with the grouping columns.
        by: Grouping columns.

    Returns:
        pl.DataFrame: ``by`` plus ``planted``, ``hit``, ``share``,
        ``low`` and ``high``.
    """
    table = truth.group_by(by).agg(planted=pl.len(), hit=pl.col("hit").sum()).sort(by)
    bounds = [wilson(h, n) for h, n in zip(table["hit"], table["planted"])]
    return table.with_columns(
        share=(pl.col("hit") / pl.col("planted")).round(3),
        low=pl.Series([round(b[0], 3) for b in bounds]),
        high=pl.Series([round(b[1], 3) for b in bounds]),
    )


def ready_hours(kind: str, inputs: Inputs) -> float:
    """Hours after a planted start at which the rule can first be met.

    Args:
        kind: Detector name.
        inputs: Detector inputs (their rules).

    Returns:
        float: The rule's least duration.
    """
    if kind == GAP:
        return inputs.gap_rules.min_hours
    if kind == CLOSURE:
        return inputs.event_rules.min_minutes / 60
    if kind == ENCOUNTER:
        return inputs.encounter_rules.min_hours
    return inputs.loiter_rules.min_hours


def delays(
    truth: pl.DataFrame, incidents: list, kind: str, ready: float
) -> pl.DataFrame:
    """Hours from when each planted positive could first alert to its alert.

    An incident belongs to a case when it is the case's detector, names its
    vessel (or, for encounters, its partner) and its evidence time falls
    between an hour before the case's start and a day after its end. A gap
    counts its first incident, which is usually the silence opened while
    the vessel is still dark.

    Args:
        truth: Matched truth (positives are scored).
        incidents: ``MicroBatchStream.incidents``.
        kind: Detector name.
        ready: ``ready_hours`` for the kind.

    Returns:
        pl.DataFrame: ``truth_id`` and ``delay_hours`` for each positive;
        null when no incident was raised.
    """
    rows = [
        {"vessel_id": i.vessel_id, "opened_at": i.opened_at, "raised_at": i.raised_at}
        for i in incidents
        if i.detector == kind and i.raised_at is not None
    ]
    raised = pl.DataFrame(
        rows,
        schema={
            "vessel_id": pl.String,
            "opened_at": pl.Datetime("us", "UTC"),
            "raised_at": pl.Datetime("us", "UTC"),
        },
    )
    positives = truth.filter(pl.col("label") == "positive")
    sides = [positives.select("truth_id", "start", "end", "vessel_id")]
    if kind == ENCOUNTER:
        sides.append(
            positives.select("truth_id", "start", "end", vessel_id="partner_id")
        )
    first = (
        pl.concat(sides)
        .join(raised, on="vessel_id")
        .filter(
            (pl.col("opened_at") >= pl.col("start") - pl.duration(hours=1))
            & (pl.col("opened_at") <= pl.col("end") + pl.duration(days=1))
        )
        .group_by("truth_id")
        .agg(pl.col("raised_at").min(), pl.col("start").first())
    )
    return positives.select("truth_id").join(
        first.select(
            "truth_id",
            delay_hours=(
                pl.col("raised_at")
                - pl.col("start")
                - pl.duration(seconds=round(ready * 3600))
            ).dt.total_seconds()
            / 3600,
        ),
        on="truth_id",
        how="left",
        maintain_order="left",
    )
