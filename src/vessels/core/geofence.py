"""Dated, gear-specific geofences: which positions fall inside a closure
that bans a gear on the day they were recorded.

A geofence table holds three kinds of rule:

- ``closure``: bans each gear in ``gear`` inside its polygon;
- ``exemption``: inside a closure (``parent``), allows each gear in
  ``gear`` again, which lifts the ban on that gear's family;
- ``suspension``: switches off an exemption (``parent``) for its dates,
  for example a fishery moratorium that leaves an exemption with no one
  entitled to use it.

Each rule is in force from ``valid_from`` (included) to ``valid_to``
(excluded, or open when None), and only inside ``season`` when one is
given. Dates are local days in the rule's ``timezone`` because
regulations are written in local dates, while AIS timestamps are UTC.

AIS does not carry gear, so callers ask about one gear family (towed
``mobile_bottom`` gear for a vessel moving at towing speed) and get every
closure that bans it, minus open exemptions for any gear in that family.
An alert is "apparent": a vessel inside an exemption may be entitled to
use it, and one inside a closure may use gear the closure allows.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
import shapely
from shapely.geometry.base import BaseGeometry

KINDS = ("closure", "exemption", "suspension")
# The family each specific gear belongs to; a family is its own family.
GEAR_FAMILY = {
    "mobile_bottom": "mobile_bottom",
    "hydraulic_clam_dredge": "mobile_bottom",
    "mussel_dredge": "mobile_bottom",
    "shrimp_trawl": "mobile_bottom",
    "sink_gillnet": "sink_gillnet",
    "demersal_longline": "demersal_longline",
}


@dataclass(frozen=True)
class Rule:  # pylint: disable=too-many-instance-attributes
    """One row of a geofence table.

    Attributes:
        area_id: Unique identifier, such as ``gsc_old_south``.
        kind: ``closure``, ``exemption`` or ``suspension``.
        gear: Gear the rule bans (closure) or allows (exemption,
            suspension).
        valid_from: First local day in force.
        valid_to: First local day no longer in force; None while open.
        geometry: Polygon in longitude and latitude.
        parent: The closure an exemption cuts out of, or the exemption a
            suspension switches off; None for a closure.
        season: Inclusive ``("MM-DD", "MM-DD")`` window each year, which
            may wrap the new year; None for all year.
        timezone: IANA zone that turns UTC timestamps into local days.
    """

    area_id: str
    kind: str
    gear: tuple[str, ...]
    valid_from: date
    valid_to: date | None
    geometry: BaseGeometry
    parent: str | None = None
    season: tuple[str, str] | None = None
    timezone: str = "America/New_York"

    def __post_init__(self):
        """Check the fields that would otherwise fail quietly later.

        Raises:
            ValueError: If the kind or a gear is unknown, the parent is
                missing or present against the kind, or the dates are
                out of order.
        """
        if self.kind not in KINDS:
            raise ValueError(f"{self.area_id}: unknown kind {self.kind!r}")
        if unknown := set(self.gear) - set(GEAR_FAMILY):
            raise ValueError(f"{self.area_id}: unknown gear {sorted(unknown)}")
        if (self.parent is None) != (self.kind == "closure"):
            raise ValueError(f"{self.area_id}: only closures have no parent")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError(f"{self.area_id}: valid_to is not after valid_from")


def check_table(rules: list[Rule]) -> None:
    """Check that a table's identifiers and parents fit together.

    Args:
        rules: The whole table.

    Raises:
        ValueError: If an identifier repeats, or a parent is missing or of
            the wrong kind (exemptions need a closure, suspensions an
            exemption).
    """
    kinds = {}
    for rule in rules:
        if rule.area_id in kinds:
            raise ValueError(f"{rule.area_id}: repeated area_id")
        kinds[rule.area_id] = rule.kind
    wanted = {"exemption": "closure", "suspension": "exemption"}
    for rule in rules:
        if rule.parent is not None and kinds.get(rule.parent) != wanted[rule.kind]:
            raise ValueError(
                f"{rule.area_id}: parent {rule.parent!r} is not of kind {wanted[rule.kind]}"
            )


def in_force(rule: Rule, day: date) -> bool:
    """Whether a rule applies on a local day.

    Args:
        rule: The rule.
        day: Local day in the rule's time zone.

    Returns:
        bool: True inside the validity dates and the season.
    """
    if day < rule.valid_from or (rule.valid_to is not None and day >= rule.valid_to):
        return False
    if rule.season is None:
        return True
    start, end = rule.season
    month_day = day.strftime("%m-%d")
    # A season such as 11-01 to 04-30 wraps the new year.
    if start <= end:
        return start <= month_day <= end
    return month_day >= start or month_day <= end


def _rules_frame(rules: list[Rule]) -> pl.DataFrame:
    """Turn rules into a frame of the fields the join needs.

    Args:
        rules: The table.

    Returns:
        pl.DataFrame: One row per rule, with ``rule`` as its list index.
    """
    return pl.DataFrame(
        {
            "rule": list(range(len(rules))),
            "area_id": [r.area_id for r in rules],
            "kind": [r.kind for r in rules],
            "parent": [r.parent for r in rules],
            "families": [sorted({GEAR_FAMILY[g] for g in r.gear}) for r in rules],
        },
        schema_overrides={"rule": pl.UInt32, "parent": pl.String},
    )


def closure_hits(
    positions: pl.DataFrame, rules: list[Rule], gear: str = "mobile_bottom"
) -> pl.DataFrame:
    """Find positions inside a closure that bans a gear family that day.

    Args:
        positions: Frame with ``timestamp`` (UTC), ``lon`` and ``lat``;
            other columns are carried through.
        rules: The geofence table.
        gear: The gear family the vessel is assumed to use.

    Returns:
        pl.DataFrame: One row per position and closure it breaks (a
        position inside two overlapping closures appears twice), with
        the position's columns plus ``closure_id`` and ``reason``:
        ``closure``, or ``suspended_exemption`` when an exemption would
        have allowed the gear but was switched off that day.

    Raises:
        ValueError: If ``gear`` is not a known family or the table does
            not pass ``check_table``.
    """
    if GEAR_FAMILY.get(gear) != gear:
        raise ValueError(f"{gear!r} is not a gear family")
    check_table(rules)
    positions = positions.with_row_index("_pos")
    empty = positions.drop("_pos").with_columns(
        closure_id=pl.lit(None, pl.String), reason=pl.lit(None, pl.String)
    )
    if not rules or positions.is_empty():
        return empty.clear()
    tree = shapely.STRtree([r.geometry for r in rules])
    points = shapely.points(positions["lon"].to_numpy(), positions["lat"].to_numpy())
    # Boundaries count as inside: a position on a closure line is in it.
    pos_idx, rule_idx = tree.query(points, predicate="covered_by")
    pairs = pl.DataFrame(
        {"_pos": pos_idx.astype(np.uint32), "rule": rule_idx.astype(np.uint32)}
    )
    days = _local_days(positions, rules, pairs)
    pairs = pairs.join(days, on=["_pos", "rule"]).filter(pl.col("in_force"))
    pairs = pairs.join(_rules_frame(rules), on="rule").filter(
        pl.col("families").list.contains(gear)
    )
    closures = pairs.filter(pl.col("kind") == "closure").select(
        "_pos", closure_id="area_id"
    )
    suspended = pairs.filter(pl.col("kind") == "suspension").select(
        "_pos", exemption="parent"
    )
    exemptions = (
        pairs.filter(pl.col("kind") == "exemption")
        .select("_pos", "area_id", closure_id="parent")
        .join(
            suspended.with_columns(suspended=pl.lit(True)),
            left_on=["_pos", "area_id"],
            right_on=["_pos", "exemption"],
            how="left",
        )
        .group_by("_pos", "closure_id")
        .agg(open=pl.col("suspended").is_null().any(), present=pl.len() > 0)
    )
    hits = (
        closures.join(exemptions, on=["_pos", "closure_id"], how="left")
        .filter(pl.col("open").fill_null(False).not_())
        .with_columns(
            reason=pl.when(pl.col("present").fill_null(False))
            .then(pl.lit("suspended_exemption"))
            .otherwise(pl.lit("closure"))
        )
        .select("_pos", "closure_id", "reason")
    )
    return (
        positions.join(hits, on="_pos")
        .sort("_pos", "closure_id")
        .drop("_pos")
        .select(empty.columns)
    )


def _local_days(
    positions: pl.DataFrame, rules: list[Rule], pairs: pl.DataFrame
) -> pl.DataFrame:
    """Decide, for each position and rule it falls in, whether the rule is
    in force on the position's local day.

    Distinct (rule, local day) combinations are few, so ``in_force`` runs
    once per combination rather than once per position.

    Args:
        positions: Positions with ``_pos`` and a UTC ``timestamp``.
        rules: The table.
        pairs: ``_pos`` and ``rule`` for each position inside a rule.

    Returns:
        pl.DataFrame: ``_pos``, ``rule`` and ``in_force``.
    """
    zones = pl.DataFrame(
        {"rule": list(range(len(rules))), "tz": [r.timezone for r in rules]},
        schema_overrides={"rule": pl.UInt32},
    )
    utc = pl.col("timestamp")
    # A timestamp without a zone is taken to be UTC, as AIS times are.
    if positions.schema["timestamp"].time_zone is None:
        utc = utc.dt.replace_time_zone("UTC")
    pairs = pairs.join(zones, on="rule").join(
        positions.select("_pos", "timestamp"), on="_pos"
    )
    local = []
    for (tz,), part in pairs.group_by("tz"):
        local.append(part.with_columns(day=utc.dt.convert_time_zone(tz).dt.date()))
    pairs = pl.concat(local) if local else pairs.with_columns(day=pl.lit(None, pl.Date))
    combos = pairs.select("rule", "day").unique()
    combos = combos.with_columns(
        in_force=pl.Series(
            [in_force(rules[r], d) for r, d in combos.iter_rows()], dtype=pl.Boolean
        )
    )
    return pairs.join(combos, on=["rule", "day"]).select("_pos", "rule", "in_force")
