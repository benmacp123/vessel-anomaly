"""Micro-batch stream: the batch detectors run on positions as they arrive.

Positions arrive out of order (``sim.arrival``). The stream holds them for
``hold`` and keeps a **watermark**: the latest timestamp seen less the
hold. Positions at or before the watermark are released in time order;
one arriving at or before a watermark already passed is late and dropped
(counted). Released positions are cleaned and segmented with state carried
between batches (``segment.Segmenter``, repeat keys, ``gaps.Moorings``),
then appended to each vessel's **tail**.

Each detector runs, as written for batch, over the tails. An episode is
**settled** when no later position can change it:

- a gap, when its end is released;
- a closure run, when the watermark is past its end by the event rules'
  split and merge times, so no later towing can join it;
- a loitering run, when a complete, not-slow hour follows it in its
  segment, or the segment has been silent for the segmenter's 24 hours;
- an encounter, when the watermark is past its end by the merge window,
  a raw encounter's least duration and the interpolation gap (6 hours),
  and past the next UTC midnight: a later
  position the same day would replace the dead-reckoned grid times after
  a vessel's last one.

Unsettled episodes that already pass their filters are opened as
incidents (``lifecycle``), and so are silences over six hours, so an
operator hears of a dark vessel while it is still dark. The watermark is
event time and moves only when positions arrive, so silences are noticed
only while other vessels are heard; a live system with no traffic would
need a wall-clock tick as well. Settling resolves them.

The tail is then trimmed to what unsettled episodes and the detectors'
look-backs still need (72 hours for gaps), always to the start of a UTC
day, because encounters resample each day from all its positions. A
trimmed tail can show a settled episode cut short; that cut-off copy
starts inside the episode already settled for the same segment and
closure (or vessel pair), and is ignored. Settled encounters that fail
their filters are remembered too, since a cut-off copy could pass.

When nothing arrives late, the stream's settled episodes equal the batch
detectors' output on the whole frame, whatever the batch boundaries
(``tests/test_stream.py``), up to the last bits of floating-point values.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import polars as pl

from vessels.core.cleaning import clean
from vessels.core.closure_fishing import DETECTOR as CLOSURE
from vessels.core.closure_fishing import EVENT_SCHEMA, closure_chains, closure_events
from vessels.core.distance import PortDistance, ShoreDistance
from vessels.core.encounters import DETECTOR as ENCOUNTER
from vessels.core.encounters import ENCOUNTER_SCHEMA, EncounterRules
from vessels.core.encounters import encounter_events, encounter_groups, settle_after
from vessels.core.events import EventRules
from vessels.core.gaps import GAP_SCHEMA, GapRules, Moorings
from vessels.core.gaps import DETECTOR as GAP
from vessels.core.gaps import gap_candidates, gap_events, gap_rows, silences
from vessels.core.geofence import Rule
from vessels.core.lifecycle import RESOLVED, Alert, AlertLifecycle, Incident
from vessels.core.loitering import DETECTOR as LOITER
from vessels.core.loitering import LOITER_SCHEMA, LoiterRules, loiter_events
from vessels.core.loitering import loiter_groups
from vessels.core.segment import MAX_HOURS, Segmenter, segment

SCHEMAS = {
    CLOSURE: EVENT_SCHEMA,
    GAP: GAP_SCHEMA,
    LOITER: LOITER_SCHEMA,
    ENCOUNTER: ENCOUNTER_SCHEMA,
}
# The batch detectors' row order, which settled episodes are sorted into.
ORDER = {
    CLOSURE: ["vessel_id", "start", "closure_id", "reason"],
    GAP: ["vessel_id", "start", "track"],
    LOITER: ["vessel_id", "start", "segment"],
    ENCOUNTER: ["vessel_id", "start", "partner_id"],
}
# How far past the last position the flush moves the watermark.
FLUSH = timedelta(days=30)
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)


@dataclass(frozen=True)
class Inputs:  # pylint: disable=too-many-instance-attributes
    """Reference layers and thresholds the detectors need.

    Attributes:
        closures: The geofence table.
        grid: Coverage grid.
        ports: Distance to anchorages (its projection is reused).
        heard: Cells and days with reception.
        shore: Distance to shore.
        event_rules: Closure event thresholds.
        gap_rules: Gap thresholds.
        loiter_rules: Loitering thresholds.
        encounter_rules: Encounter thresholds.
    """

    closures: list[Rule]
    grid: pl.DataFrame
    ports: PortDistance
    heard: pl.DataFrame
    shore: ShoreDistance
    event_rules: EventRules = field(default_factory=EventRules)
    gap_rules: GapRules = field(default_factory=GapRules)
    loiter_rules: LoiterRules = field(default_factory=LoiterRules)
    encounter_rules: EncounterRules = field(default_factory=EncounterRules)


def run_batch(frame: pl.DataFrame, inputs: Inputs) -> dict[str, pl.DataFrame]:
    """Run every detector over a whole frame at once.

    Args:
        frame: Positions in the extract schema.
        inputs: Reference layers and thresholds.

    Returns:
        dict[str, pl.DataFrame]: Events by detector name.
    """
    segmented = segment(clean(frame).frame)
    return {
        CLOSURE: closure_events(segmented, inputs.closures, inputs.event_rules),
        GAP: gap_events(
            segmented, inputs.grid, inputs.ports, inputs.heard, inputs.gap_rules
        ),
        LOITER: loiter_events(segmented, inputs.shore, inputs.loiter_rules),
        ENCOUNTER: encounter_events(
            segmented, inputs.ports, inputs.shore, inputs.encounter_rules
        ),
    }


class MicroBatchStream:  # pylint: disable=too-many-instance-attributes
    """Feeds arriving positions through the detectors a batch at a time.

    Attributes:
        inputs: Reference layers and thresholds.
        hold: Reorder hold.
        counts: Positions received, dropped late and dropped as duplicates.
        incidents: Every incident transition, in order.
    """

    def __init__(self, inputs: Inputs, hold: timedelta):
        """Start an empty stream.

        Args:
            inputs: Reference layers and thresholds.
            hold: Reorder hold.
        """
        self.inputs = inputs
        self.hold = hold
        self.counts = {"received": 0, "late": 0, "duplicate": 0}
        self.incidents: list[Incident] = []
        self._buffer: pl.DataFrame | None = None
        self._tail: pl.DataFrame | None = None
        self._flags: pl.DataFrame | None = None
        self._latest: datetime | None = None
        self._released: datetime | None = None
        self._segmenter = Segmenter()
        self._seen: dict = {}
        self._moorings = Moorings(inputs.ports.projection)
        self._lifecycle = AlertLifecycle()
        self._states: dict[str, dict] = {}
        self._emitted_until: dict[tuple, datetime] = {}
        self._settled: dict[str, list[pl.DataFrame]] = {d: [] for d in SCHEMAS}
        # Unsettled episodes from the last detection pass, for trimming.
        self._pending_closure: pl.DataFrame | None = None
        self._pending_loiter: tuple | None = None
        self._pending_encounter: pl.DataFrame | None = None

    def push(self, arrivals: pl.DataFrame, now: datetime) -> list[Incident]:
        """Take one batch of arrivals and release what the watermark allows.

        Args:
            arrivals: Positions in the extract schema; an ``arrival`` column
                is ignored.
            now: Processing time, recorded on incidents.

        Returns:
            list[Incident]: Incident transitions raised by this batch.
        """
        arrivals = arrivals.drop("arrival", strict=False)
        self.counts["received"] += arrivals.height
        if self._released is not None:
            late = arrivals["timestamp"] <= self._released
            self.counts["late"] += int(late.sum())
            arrivals = arrivals.filter(late.not_())
        if arrivals.is_empty():
            return []
        self._buffer = (
            arrivals if self._buffer is None else pl.concat([self._buffer, arrivals])
        )
        latest = arrivals["timestamp"].max()
        self._latest = latest if self._latest is None else max(self._latest, latest)
        watermark = self._latest - self.hold
        if self._released is not None and watermark <= self._released:
            return []
        return self._advance(watermark, now, final=False)

    def flush(self, now: datetime) -> list[Incident]:
        """Release everything held and settle every episode.

        Args:
            now: Processing time.

        Returns:
            list[Incident]: Incident transitions raised.
        """
        if self._latest is None:
            return []
        return self._advance(self._latest + FLUSH, now, final=True)

    def drain(self) -> tuple[dict[str, pl.DataFrame], list[Incident]]:
        """Hand over settled episodes and incident transitions, and forget
        them, so a long-running stream's state does not grow with its
        output.

        Returns:
            tuple: ``events`` output and the incident transitions since the
            last drain.
        """
        events, incidents = self.events(), self.incidents
        self._settled = {d: [] for d in SCHEMAS}
        self.incidents = []
        return events, incidents

    def events(self) -> dict[str, pl.DataFrame]:
        """Settled episodes since the last drain, in the batch detectors' order.

        Returns:
            dict[str, pl.DataFrame]: Events by detector name.
        """
        return {
            name: (
                pl.concat(parts).select(list(SCHEMAS[name])).sort(ORDER[name])
                if parts
                else pl.DataFrame(schema=SCHEMAS[name])
            )
            for name, parts in self._settled.items()
        }

    def _advance(self, watermark: datetime, now: datetime, final: bool) -> list:
        """Release positions up to the watermark, detect and trim.

        Args:
            watermark: New watermark.
            now: Processing time.
            final: True at the flush: settle everything.

        Returns:
            list[Incident]: Incident transitions raised.
        """
        ready = self._buffer.filter(pl.col("timestamp") <= watermark)
        self._buffer = self._buffer.filter(pl.col("timestamp") > watermark)
        previous, self._released = self._released, watermark
        if not ready.is_empty():
            self._ingest(ready, previous)
        if self._tail is None:
            return []
        alerts = self._gaps(watermark, previous, final)
        alerts += self._closures(watermark, final)
        alerts += self._loitering(watermark, final)
        alerts += self._encounters(watermark, final)
        found = []
        for alert in alerts:
            state = self._states.setdefault(alert.vessel_id, {})
            found += self._lifecycle.apply(state, [alert], now)
        self.incidents += found
        self._trim(watermark)
        return found

    def _ingest(self, ready: pl.DataFrame, previous: datetime | None) -> None:
        """Clean, segment and append released positions; check moorings for
        gaps they end.

        Args:
            ready: Released positions.
            previous: The watermark before this release, or None.
        """
        result = clean(ready)
        self.counts["duplicate"] += result.dropped["duplicate"]
        labelled = segment(result.frame, self._segmenter, self._seen)
        self._tail = (
            labelled if self._tail is None else pl.concat([self._tail, labelled])
        )
        found = gap_candidates(self._tail, self.inputs.gap_rules)
        if previous is not None:
            found = found.filter(pl.col("end") > previous)
        flags = self._moorings.flags(
            gap_rows(labelled), found, self.inputs.gap_rules.mooring_km
        )
        self._flags = flags if self._flags is None else pl.concat([self._flags, flags])

    def _is_open(self, vessel: str, episode: str) -> bool:
        """Whether an episode has an open incident.

        Args:
            vessel: Vessel ID.
            episode: Episode ID.

        Returns:
            bool: True if open.
        """
        return episode in self._states.get(vessel, {}).get("open", {})

    def _open_ids(self, detector: str) -> list[tuple[str, str]]:
        """Open episodes of one detector.

        Args:
            detector: Detector name.

        Returns:
            list[tuple[str, str]]: ``(vessel, episode)`` pairs.
        """
        return [
            (vessel, episode)
            for vessel, state in self._states.items()
            for episode, record in state.get("open", {}).items()
            if record["detector"] == detector
        ]

    def _settle(self, detector: str, key: tuple, row: dict) -> list[Alert]:
        """Record a settled episode and resolve its incident, opening it
        first if it never was.

        Args:
            detector: Detector name.
            key: The segment (and closure) the episode belongs to.
            row: The episode.

        Returns:
            list[Alert]: Transitions to apply.
        """
        self._settled[detector].append(pl.DataFrame([row], schema=SCHEMAS[detector]))
        self._emitted_until[key] = row["end"]
        vessel, episode = row["vessel_id"], row["event_id"]
        alerts = []
        if not self._is_open(vessel, episode):
            alerts.append(Alert(vessel, detector, episode, row["start"], detector))
        reason = "final"
        if detector == GAP:
            reason = "unexplained" if row["unexplained"] else "explained"
        alerts.append(
            Alert(vessel, detector, episode, row["end"], detector, RESOLVED, reason)
        )
        return alerts

    def _gaps(self, watermark, previous, final: bool) -> list[Alert]:
        """Settle gaps that ended in this release and open new silences.

        Args:
            watermark: Current watermark.
            previous: Previous watermark, or None.
            final: True at the flush.

        Returns:
            list[Alert]: Transitions to apply.
        """
        rules = self.inputs.gap_rules
        found = gap_events(
            self._tail,
            self.inputs.grid,
            self.inputs.ports,
            self.inputs.heard,
            rules,
            mooring=self._flags,
        )
        if previous is not None:
            found = found.filter(pl.col("end") > previous)
        alerts = []
        for row in found.iter_rows(named=True):
            alerts += self._settle(GAP, (GAP, row["track"]), row)
        ended = set(found["event_id"])
        if final:
            return alerts
        quiet = silences(self._tail, watermark, rules)
        for row in quiet.iter_rows(named=True):
            if not self._is_open(row["vessel_id"], row["event_id"]):
                alerts.append(
                    Alert(
                        row["vessel_id"],
                        GAP,
                        row["event_id"],
                        row["start"] + timedelta(hours=rules.min_hours),
                        "silent",
                    )
                )
        still = set(quiet["event_id"]) | ended
        for vessel, episode in self._open_ids(GAP):
            if episode not in still:
                alerts.append(
                    Alert(
                        vessel, GAP, episode, watermark, "closed", RESOLVED, "withdrawn"
                    )
                )
        return alerts

    def _closures(self, watermark, final: bool) -> list[Alert]:
        """Open passing closure runs and settle those that can no longer
        grow.

        Args:
            watermark: Current watermark.
            final: True at the flush.

        Returns:
            list[Alert]: Transitions to apply.
        """
        rules = self.inputs.event_rules
        horizon = timedelta(hours=max(rules.split_hours, rules.merge_hours))
        found = closure_chains(self._tail, self.inputs.closures, rules)
        found = found.with_columns(
            settled=pl.lit(final) | (pl.col("end") <= watermark - horizon)
        )
        self._pending_closure = found.filter(pl.col("settled").not_())
        alerts = []
        for row in found.filter("passes").iter_rows(named=True):
            key = (CLOSURE, row["segment"], row["closure_id"], row["reason"])
            if key in self._emitted_until and row["start"] <= self._emitted_until[key]:
                continue
            if row["settled"]:
                alerts += self._settle(CLOSURE, key, row)
            elif not self._is_open(row["vessel_id"], row["event_id"]):
                alerts.append(
                    Alert(
                        row["vessel_id"],
                        CLOSURE,
                        row["event_id"],
                        row["start"],
                        CLOSURE,
                    )
                )
        return alerts

    def _loitering(self, watermark, final: bool) -> list[Alert]:
        """Open passing loitering runs, settle finished ones and withdraw
        those that stopped passing.

        Args:
            watermark: Current watermark.
            final: True at the flush.

        Returns:
            list[Alert]: Transitions to apply.
        """
        groups, buckets = loiter_groups(
            self._tail, self.inputs.shore, self.inputs.loiter_rules
        )
        closers = buckets.filter(
            pl.col("slow").not_(), pl.col("bucket") + HOUR <= watermark
        ).select("segment", close="bucket")
        last = buckets.group_by("segment").agg(seg_last=pl.col("last_time").max())
        closed = (
            groups.select(
                "event_id", "segment", end_bucket=pl.col("end").dt.truncate("1h")
            )
            .join(closers, on="segment", how="left")
            .group_by("event_id")
            .agg(closed=(pl.col("close") > pl.col("end_bucket")).fill_null(False).any())
        )
        groups = (
            groups.join(closed, on="event_id", how="left", maintain_order="left")
            .join(last, on="segment", how="left", maintain_order="left")
            .with_columns(
                settled=pl.lit(final)
                | pl.col("closed")
                | (pl.lit(watermark) - pl.col("seg_last") > timedelta(hours=MAX_HOURS))
            )
        )
        self._pending_loiter = (groups.filter(pl.col("settled").not_()), buckets, last)
        alerts = []
        passing = set()
        for row in groups.filter("passes").iter_rows(named=True):
            key = (LOITER, row["segment"])
            if key in self._emitted_until and row["start"] <= self._emitted_until[key]:
                continue
            passing.add(row["event_id"])
            if row["settled"]:
                alerts += self._settle(LOITER, key, row)
            elif not self._is_open(row["vessel_id"], row["event_id"]):
                alerts.append(
                    Alert(
                        row["vessel_id"], LOITER, row["event_id"], row["start"], LOITER
                    )
                )
        for vessel, episode in self._open_ids(LOITER):
            if episode not in passing:
                alerts.append(
                    Alert(
                        vessel,
                        LOITER,
                        episode,
                        watermark,
                        LOITER,
                        RESOLVED,
                        "withdrawn",
                    )
                )
        return alerts

    def _encounters(self, watermark, final: bool) -> list[Alert]:
        """Open passing encounters, settle finished ones and withdraw those
        that stopped passing.

        Args:
            watermark: Current watermark.
            final: True at the flush.

        Returns:
            list[Alert]: Transitions to apply.
        """
        rules = self.inputs.encounter_rules
        found = encounter_groups(
            self._tail, self.inputs.ports, self.inputs.shore, rules
        ).with_columns(
            settled=pl.lit(final)
            | (
                (pl.col("end") + settle_after(rules) <= watermark)
                & (pl.col("end").dt.truncate("1d") + DAY <= watermark)
            )
        )
        self._pending_encounter = found.filter(pl.col("settled").not_())
        alerts = []
        passing = set()
        for row in found.iter_rows(named=True):
            key = (ENCOUNTER, row["vessel_id"], row["partner_id"])
            if key in self._emitted_until and row["start"] <= self._emitted_until[key]:
                continue
            if row["settled"] and row["passes"]:
                alerts += self._settle(ENCOUNTER, key, row)
            elif row["settled"]:
                self._emitted_until[key] = row["end"]
            elif row["passes"]:
                passing.add(row["event_id"])
                if not self._is_open(row["vessel_id"], row["event_id"]):
                    alerts.append(
                        Alert(
                            row["vessel_id"],
                            ENCOUNTER,
                            row["event_id"],
                            row["start"],
                            ENCOUNTER,
                        )
                    )
        settled = set(found.filter("settled", "passes")["event_id"])
        for vessel, episode in self._open_ids(ENCOUNTER):
            if episode not in passing and episode not in settled:
                alerts.append(
                    Alert(
                        vessel,
                        ENCOUNTER,
                        episode,
                        watermark,
                        ENCOUNTER,
                        RESOLVED,
                        "withdrawn",
                    )
                )
        return alerts

    def _trim(self, watermark: datetime) -> None:
        """Drop tail positions no unsettled episode or look-back needs.

        Args:
            watermark: Current watermark.
        """
        # Three segmenter days: a shared-ID gap starts up to a day before
        # it ends, and its start needs a day of history for the shared
        # check, whose own rows need a day more.
        lookback = timedelta(hours=3 * MAX_HOURS)
        keeps = [
            self._tail.group_by("vessel_id").agg(
                keep=pl.col("timestamp").max() - lookback
            )
        ]
        keeps.append(
            self._pending_closure.group_by("vessel_id").agg(keep=pl.col("start").min())
        )
        pending, buckets, last = self._pending_loiter
        # A loitering run needs the hour before its lead-in hour intact;
        # an open segment's latest hour may lead into a future run.
        keeps.append(
            pending.group_by("vessel_id").agg(
                keep=pl.col("start").dt.truncate("1h").min() - HOUR
            )
        )
        live = last.filter(
            pl.lit(watermark) - pl.col("seg_last") <= timedelta(hours=MAX_HOURS)
        )
        latest = (
            buckets.join(live, on="segment")
            .group_by("segment")
            .agg(keep=pl.col("bucket").max() - HOUR)
        )
        owners = self._tail.select("segment", "vessel_id").unique()
        keeps.append(latest.join(owners, on="segment").select("vessel_id", "keep"))
        # An unsettled encounter needs both vessels from its start.
        pending = self._pending_encounter
        for side in ("vessel_id", "partner_id"):
            keeps.append(
                pending.group_by(side)
                .agg(keep=pl.col("start").min())
                .rename({side: "vessel_id"})
            )
        keep = (
            pl.concat([k.select("vessel_id", "keep") for k in keeps])
            .group_by("vessel_id")
            .agg(pl.col("keep").min().dt.truncate("1d"))
        )
        self._tail = (
            self._tail.join(keep, on="vessel_id", how="left", maintain_order="left")
            .filter(pl.col("timestamp") >= pl.col("keep"))
            .drop("keep")
        )
        self._flags = (
            self._flags.join(keep, on="vessel_id", how="left", maintain_order="left")
            .filter(pl.col("start") >= pl.col("keep"))
            .drop("keep")
        )


def batches(arrived: pl.DataFrame, every: timedelta) -> Iterator[tuple]:
    """Split arrivals into batches of ``every`` by arrival time.

    Args:
        arrived: Positions with an ``arrival`` column (``sim.arrival``).
        every: Batch length.

    Yields:
        tuple: The batch's processing time (the end of its window) and its
        positions, in arrival order.
    """
    marked = arrived.with_columns(_batch=pl.col("arrival").dt.truncate(every))
    for (start,), part in marked.sort("arrival", maintain_order=True).group_by(
        "_batch", maintain_order=True
    ):
        yield start + every, part.drop("_batch")


def replay(
    arrived: pl.DataFrame, inputs: Inputs, hold: timedelta, every: timedelta
) -> MicroBatchStream:
    """Feed arrivals to a stream in batches of ``every`` by arrival time.

    Args:
        arrived: Positions with an ``arrival`` column (``sim.arrival``).
        inputs: Reference layers and thresholds.
        hold: Reorder hold.
        every: Batch length.

    Returns:
        MicroBatchStream: The stream after the flush.
    """
    stream = MicroBatchStream(inputs, hold)
    for now, part in batches(arrived, every):
        stream.push(part, now)
    if arrived.height:
        stream.flush(now=arrived["arrival"].max() + every)
    return stream
