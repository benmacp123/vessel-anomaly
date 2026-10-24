"""Apparent fishing inside a closure: lasting towing runs in a closure
that bans towed gear on the day.

A position is "towing" when a fishing-type vessel reports 2.5-5.5 kn, the
trawl speed band of de Souza et al. 2016 (84% of trawl positions;
literature-sourced). AIS carries no gear, so every towing position is
assumed to be mobile bottom-tending gear; gillnetters and lobster boats
hauling at similar speeds are false positives the rule cannot avoid.

For each closure and reason (``closure`` or ``suspended_exemption``, from
``geofence.closure_hits``) the state "towing inside this closure" runs
through GFW's fishing-event rules (``events.flag_events``) along each
segment: brief exits and speed changes within an hour and 2 km merge, and
events shorter than 20 minutes or 0.5 km are dropped. Every result is
"apparent": the vessel may use gear the closure allows, or hold an
exemption it is entitled to.
"""

import polars as pl

from vessels.core.events import EventRules, chains, steps
from vessels.core.geofence import Rule, closure_hits

DETECTOR = "closure_fishing"
# Literature-sourced (de Souza et al. 2016): trawl speed band, knots.
TOWING_KN = (2.5, 5.5)

EVENT_SCHEMA = {
    "event_id": pl.String,
    "vessel_id": pl.String,
    "segment": pl.String,
    "closure_id": pl.String,
    "reason": pl.String,
    "transceiver": pl.String,
    "start": pl.Datetime("us", "UTC"),
    "end": pl.Datetime("us", "UTC"),
    "hours": pl.Float64,
    "positions": pl.UInt32,
    "distance_km": pl.Float64,
    "avg_sog": pl.Float64,
    "start_lon": pl.Float64,
    "start_lat": pl.Float64,
    "end_lon": pl.Float64,
    "end_lat": pl.Float64,
}


def closure_events(
    segmented: pl.DataFrame,
    rules: list[Rule],
    event_rules: EventRules = EventRules(),
    towing_kn: tuple[float, float] = TOWING_KN,
) -> pl.DataFrame:
    """Find lasting towing runs inside closures.

    Args:
        segmented: ``segment.segment`` output; only rows with a segment
            are used.
        rules: The geofence table.
        event_rules: Event thresholds (GFW's by default).
        towing_kn: Inclusive speed band counted as towing.

    Returns:
        pl.DataFrame: The ``closure_chains`` that pass, in ``EVENT_SCHEMA``.
    """
    found = closure_chains(segmented, rules, event_rules, towing_kn)
    return found.filter("passes").drop("passes")


def closure_chains(
    segmented: pl.DataFrame,
    rules: list[Rule],
    event_rules: EventRules = EventRules(),
    towing_kn: tuple[float, float] = TOWING_KN,
) -> pl.DataFrame:
    """Towing runs inside closures, whether or not they last long enough.

    Args:
        segmented: ``segment.segment`` output; only rows with a segment
            are used.
        rules: The geofence table.
        event_rules: Event thresholds (GFW's by default).
        towing_kn: Inclusive speed band counted as towing.

    Returns:
        pl.DataFrame: One row per run in ``EVENT_SCHEMA`` plus ``passes``,
        ordered by vessel, start and closure. ``event_id`` is the segment,
        closure, reason and start time, so a rerun gives the same IDs.
    """
    tracks = (
        segmented.filter(pl.col("segment").is_not_null())
        .sort("segment", "timestamp", maintain_order=True)
        .pipe(steps, "segment")
        .with_row_index("_row")
    )
    towing = tracks.filter(pl.col("sog").is_between(*towing_kn))
    hits = closure_hits(
        towing.select("_row", "timestamp", "lon", "lat"), rules, gear="mobile_bottom"
    )
    found = []
    for (closure_id, reason), part in hits.group_by(
        "closure_id", "reason", maintain_order=True
    ):
        rows = part["_row"]
        # Only segments with a hit can hold an event; the rest are skipped.
        segments = tracks.filter(pl.col("_row").is_in(rows.implode()))[
            "segment"
        ].unique()
        track = tracks.filter(pl.col("segment").is_in(segments.implode())).with_columns(
            _flag=pl.col("_row").is_in(rows.implode())
        )
        events = chains(track, "_flag", "segment", event_rules)
        found.append(
            events.with_columns(
                closure_id=pl.lit(closure_id), reason=pl.lit(reason)
            ).drop("event")
        )
    schema = {**EVENT_SCHEMA, "passes": pl.Boolean}
    if not found:
        return pl.DataFrame(schema=schema)
    meta = tracks.group_by("segment").agg(
        pl.col("vessel_id").first(), pl.col("transceiver").first()
    )
    return (
        pl.concat(found)
        .join(meta, on="segment")
        .with_columns(
            event_id=pl.concat_str(
                "segment",
                "closure_id",
                "reason",
                pl.col("start").dt.strftime("%Y%m%dT%H%M%S"),
                separator=":",
            )
        )
        .select(pl.col(name).cast(dtype) for name, dtype in schema.items())
        .sort("vessel_id", "start", "closure_id", "reason")
    )
