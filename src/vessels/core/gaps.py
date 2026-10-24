"""AIS gaps, and which of them are unexplained inside receiver coverage.

A gap is the silence between two consecutive positions lasting over
``min_hours``. Every rule uses only what was known when the gap ended, so
a stream fed the same positions finds the same gaps (``stream``).

**Which positions are consecutive.** Usually the vessel's own: the
segmenter ends a segment after 24 hours of silence, so the longest gaps
fall between segments. When the vessel had another segment heard in the
24 hours before the gap began, two transmitters probably share its ID and
their positions interleave, hiding each one's silence; then consecutive
positions are taken within the segment instead. Moving repeats are left
out, since a relayed copy of an old message proves nothing about the
transmitter; ``bad`` rows (no speed) are kept, since the vessel was heard,
but belong to no segment.

**Unexplained.** Most silence is reception or harbour, not intent. Welch
et al. 2022 call a gap likely intentional when it lasts 12 hours or more,
starts over 50 nm offshore, in a cell with good reception, after at least
14 positions in the previous 12 hours. With terrestrial US data almost
nothing lies 50 nm out, so here a gap is unexplained when all of these
hold (provisional, owner-agreed 2026-10-24):

- ``covered``: both ends lie in H3 cells covered for their AIS class
  (``coverage.coverage``);
- ``away_from_port``: both ends are ``min_port_km`` or more from an
  anchorage, since vessels in harbour switch off or drop out of range;
- ``away_from_mooring``: neither end lies within ``mooring_km`` of the
  vessel's own mooring (``Moorings``). Anchorages need two vessels, so a
  boat's home mooring is often not one: on the tuning block 43 of 157
  otherwise unexplained gaps started stationary on land or within 2 km of
  shore (observed, 2026-10-24);
- ``dense_before``: at least ``min_prior_positions`` positions of the same
  track in the ``prior_hours`` up to the gap's start (Welch);
- ``cells_heard``: both end cells heard traffic of any class on every UTC
  day the gap spans. The static coverage mask cannot see a receiver that
  was down for a day, and daily is the finest check the committed
  reception grid allows. A gap spanning a day outside the data (such as
  the months between the May and September halves of a block) fails it.
  This is the one input a live system would not have until the day ends.

``long`` marks unexplained gaps of ``long_hours`` or more, Welch's
duration. Gaps still open at the end of the data are not reported.
"""

from dataclasses import dataclass

import h3
import numpy as np
import polars as pl

from vessels.core.distance import MIN_HOURS, RADIUS_KM, PortDistance
from vessels.core.events import steps
from vessels.core.geo import LocalProjection, haversine_km
from vessels.core.segment import MAX_HOURS

DETECTOR = "gap"
MICROS_PER_HOUR = 3_600_000_000


@dataclass(frozen=True)
class GapRules:
    """Thresholds for recording a gap and calling it unexplained.

    Attributes:
        min_hours: A gap must last longer than this (GFW records gaps over
            6 hours; literature-sourced).
        long_hours: Unexplained gaps at least this long are ``long``
            (Welch et al. 2022; literature-sourced).
        prior_hours: Window before the gap for counting positions (Welch).
        min_prior_positions: Positions needed in that window (Welch).
        min_port_km: Distance from an anchorage both ends need
            (provisional).
        mooring_km: Distance from the vessel's own mooring both ends need
            (provisional).
        shared_hours: Another segment heard this recently before a gap
            means a shared ID (the segmenter's 24-hour limit).
    """

    min_hours: float = 6.0
    long_hours: float = 12.0
    prior_hours: float = 12.0
    min_prior_positions: int = 14
    min_port_km: float = 10.0
    mooring_km: float = 1.0
    shared_hours: float = MAX_HOURS


GAP_SCHEMA = {
    "event_id": pl.String,
    "vessel_id": pl.String,
    "track": pl.String,
    "transceiver": pl.String,
    "start": pl.Datetime("us", "UTC"),
    "end": pl.Datetime("us", "UTC"),
    "hours": pl.Float64,
    "distance_km": pl.Float64,
    "implied_kn": pl.Float64,
    "start_lon": pl.Float64,
    "start_lat": pl.Float64,
    "end_lon": pl.Float64,
    "end_lat": pl.Float64,
    "start_cell": pl.String,
    "end_cell": pl.String,
    "prior_positions": pl.UInt32,
    "start_port_km": pl.Float64,
    "end_port_km": pl.Float64,
    "covered": pl.Boolean,
    "away_from_port": pl.Boolean,
    "away_from_mooring": pl.Boolean,
    "dense_before": pl.Boolean,
    "cells_heard": pl.Boolean,
    "unexplained": pl.Boolean,
    "long": pl.Boolean,
}
# The candidate columns ``Moorings.flags`` needs and returns with.
CANDIDATE_KEY = ["vessel_id", "track", "start", "end"]


class Moorings:
    """Where each vessel has been heard stationary for long, built up one
    position at a time.

    A vessel is stationary while it stays within ``RADIUS_KM`` of the first
    position of its current stay, the rule ``distance.stationary_runs``
    uses (GFW's); leaving starts a new stay at the position that left. A
    stay that lasted ``MIN_HOURS`` or more, first to last position, is a
    mooring at the mean of its positions; so is the current stay once it has
    lasted that long. A gap is checked before its end position is added, so
    it cannot explain itself by reappearing where it fell silent, and later
    stays never explain an earlier gap.

    State is plain JSON (``state``), so a stream can save and resume it.

    Attributes:
        projection: Projection for the stationary radius.
        state: Per vessel: the current stay and the moorings so far.
    """

    def __init__(self, projection: LocalProjection, state: dict | None = None):
        """Start empty or resume.

        Args:
            projection: Projection about the study area.
            state: A previous ``state``, or None to start empty.
        """
        self.projection = projection
        self.state = state if state is not None else {}

    def add(self, vessel: str, micros: int, lon: float, lat: float) -> None:
        """Add one position, in time order for its vessel.

        Args:
            vessel: Vessel ID.
            micros: Timestamp, microseconds since the epoch.
            lon: Longitude.
            lat: Latitude.
        """
        x, y = self.projection.forward(np.array([lon]), np.array([lat]))
        x, y = float(x[0]), float(y[0])
        stay = self.state.setdefault(vessel, {"stay": None, "sites": []})
        current = stay["stay"]
        if current is not None:
            ax, ay = current["anchor"]
            if (x - ax) ** 2 + (y - ay) ** 2 <= RADIUS_KM**2:
                current["last"] = micros
                current["lon"] += lon
                current["lat"] += lat
                current["n"] += 1
                return
            if self._long(current):
                stay["sites"].append(self._centre(current))
        stay["stay"] = {
            "anchor": [x, y],
            "first": micros,
            "last": micros,
            "lon": lon,
            "lat": lat,
            "n": 1,
        }

    @staticmethod
    def _long(stay: dict) -> bool:
        """Whether a stay lasted long enough to be a mooring.

        Args:
            stay: A stay record.

        Returns:
            bool: True if it lasted ``MIN_HOURS`` or more.
        """
        return stay["last"] - stay["first"] >= MIN_HOURS * MICROS_PER_HOUR

    @staticmethod
    def _centre(stay: dict) -> list[float]:
        """Mean position of a stay.

        Args:
            stay: A stay record.

        Returns:
            list[float]: ``[lon, lat]``.
        """
        return [stay["lon"] / stay["n"], stay["lat"] / stay["n"]]

    def near(self, vessel: str, points: list[tuple[float, float]], km: float) -> bool:
        """Whether any point lies within ``km`` of one of the vessel's
        moorings so far.

        Args:
            vessel: Vessel ID.
            points: ``(lon, lat)`` pairs.
            km: Distance limit.

        Returns:
            bool: True if any point is near a mooring.
        """
        record = self.state.get(vessel)
        if record is None:
            return False
        sites = list(record["sites"])
        if record["stay"] is not None and self._long(record["stay"]):
            sites.append(self._centre(record["stay"]))
        if not sites:
            return False
        lons, lats = np.array(sites).T
        return any(
            bool((haversine_km(lon, lat, lons, lats) <= km).any())
            for lon, lat in points
        )

    def flags(
        self, rows: pl.DataFrame, candidates: pl.DataFrame, km: float
    ) -> pl.DataFrame:
        """Add positions in order, checking each gap just before its end.

        Args:
            rows: ``vessel_id``, ``timestamp``, ``lon``, ``lat``, sorted by
                vessel then time: every position after those already added.
            candidates: Gaps ending among ``rows``, with ``CANDIDATE_KEY``
                and the end coordinates.
            km: Mooring distance limit.

        Returns:
            pl.DataFrame: ``CANDIDATE_KEY`` plus ``away_from_mooring``.
        """
        due: dict[tuple[str, int], list[tuple]] = {}
        for gap in candidates.iter_rows(named=True):
            key = (gap["vessel_id"], _micros(gap["end"]))
            due.setdefault(key, []).append(gap)
        out = []
        times = (rows["timestamp"].dt.epoch("us")).to_list()
        for vessel, micros, lon, lat in zip(
            rows["vessel_id"].to_list(), times, rows["lon"], rows["lat"]
        ):
            for gap in due.pop((vessel, micros), []):
                points = [
                    (gap["start_lon"], gap["start_lat"]),
                    (gap["end_lon"], gap["end_lat"]),
                ]
                out.append(
                    {
                        **{k: gap[k] for k in CANDIDATE_KEY},
                        "away_from_mooring": not self.near(vessel, points, km),
                    }
                )
            self.add(vessel, micros, lon, lat)
        schema = {k: candidates.schema[k] for k in CANDIDATE_KEY}
        return pl.DataFrame(out, schema={**schema, "away_from_mooring": pl.Boolean})


def _micros(time) -> int:
    """Microseconds since the epoch for a timezone-aware datetime.

    Args:
        time: A UTC datetime.

    Returns:
        int: Microseconds.
    """
    return pl.Series([time]).dt.epoch("us")[0]


def gap_rows(segmented: pl.DataFrame) -> pl.DataFrame:
    """The positions gaps are measured between, in time order per vessel.

    Args:
        segmented: ``segment.segment`` output.

    Returns:
        pl.DataFrame: Rows other than repeats, sorted by vessel then time
        (ties keep their input order).
    """
    return (
        segmented.filter(pl.col("point_status") != "repeat")
        .select("vessel_id", "timestamp", "lon", "lat", "transceiver", "segment")
        .sort("vessel_id", "timestamp", maintain_order=True)
    )


def _shared(rows: pl.DataFrame, hours: float) -> pl.Series:
    """Whether another segment of the vessel was heard in the window up to
    each position.

    Only rows with a segment count as heard on a segment. A row without
    one takes the flag of the vessel's last row with a segment, if that
    row is inside the window; otherwise it is not shared. Both rules look
    back at most ``hours``, so a stream holding that much history gets the
    same flags.

    Args:
        rows: ``gap_rows`` output.
        hours: Window length.

    Returns:
        pl.Series: One flag per row.
    """
    window = hours * MICROS_PER_HOUR
    indexed = rows.with_row_index("_i")
    heard = indexed.filter(pl.col("segment").is_not_null()).with_columns(
        _run=(pl.col("segment") != pl.col("segment").shift(1))
        .fill_null(True)
        .cum_sum()
        .over("vessel_id")
    )
    # Each run of one segment is paired with the run before it: the last
    # time a different segment of the vessel was heard.
    before = heard.group_by("vessel_id", "_run").agg(_end=pl.col("timestamp").last())
    before = before.with_columns(_run=pl.col("_run") + 1)
    since = (pl.col("timestamp") - pl.col("_end")).dt.total_microseconds()
    heard = heard.join(
        before, on=["vessel_id", "_run"], how="left", maintain_order="left"
    ).select("_i", _flag=(since <= window).fill_null(False), _at="timestamp")
    filled = (
        indexed.join(heard, on="_i", how="left", maintain_order="left")
        .with_columns(
            pl.col("_flag").forward_fill().over("vessel_id"),
            pl.col("_at").forward_fill().over("vessel_id"),
        )
        .select(
            shared=(
                pl.col("_flag")
                & (
                    (pl.col("timestamp") - pl.col("_at")).dt.total_microseconds()
                    <= window
                )
            ).fill_null(False)
        )
    )
    return filled["shared"]


def _count_before(tracks: pl.DataFrame, hours: float) -> np.ndarray:
    """Positions of the same track in the window up to each position.

    Args:
        tracks: Rows sorted by ``track`` and time.
        hours: Window length; both ends are included.

    Returns:
        np.ndarray: Count per row, including the row itself.
    """
    if tracks.is_empty():
        return np.empty(0, dtype=np.uint32)
    rank = tracks["track"].ne(tracks["track"].shift(1)).fill_null(True).cum_sum()
    micros = (tracks["timestamp"] - tracks["timestamp"].min()).dt.total_microseconds()
    # One sortable key per row: the track's rank, then time within it. A
    # block spans under 2**46 microseconds (about two years).
    key = rank.cast(pl.Int64).to_numpy() * 2**46 + micros.to_numpy()
    first = np.searchsorted(key, key - round(hours * MICROS_PER_HOUR), side="left")
    return (np.arange(key.size) - first + 1).astype(np.uint32)


def gap_candidates(
    segmented: pl.DataFrame, rules: GapRules = GapRules()
) -> pl.DataFrame:
    """Every silence over ``min_hours`` between consecutive positions.

    Args:
        segmented: ``segment.segment`` output.
        rules: Thresholds.

    Returns:
        pl.DataFrame: ``CANDIDATE_KEY``, ``transceiver`` (at the start),
        ``end_transceiver``, ``hours``, ``distance_km``, the end
        coordinates and ``prior_positions``.
    """
    rows = gap_rows(segmented)
    rows = rows.with_columns(_shared=_shared(rows, rules.shared_hours))
    vessel = rows.with_columns(track=pl.col("vessel_id"))
    per_segment = (
        rows.filter(pl.col("segment").is_not_null())
        .with_columns(track=pl.col("segment"))
        .sort("track", "timestamp", maintain_order=True)
    )
    found = []
    # A gap starting while the vessel looked shared is measured within the
    # segment; otherwise across the vessel's positions.
    for tracks, shared in ((vessel, False), (per_segment, True)):
        tracks = tracks.pipe(steps, "track").with_columns(
            prior_positions=pl.Series(
                _count_before(tracks, rules.prior_hours), dtype=pl.UInt32
            )
        )
        previous = [
            pl.col(c).shift(1).alias(f"start_{c}")
            for c in ("timestamp", "lon", "lat", "transceiver", "prior_positions")
        ]
        found.append(
            tracks.with_columns(previous, start_shared=pl.col("_shared").shift(1))
            .filter(
                pl.col("hours_to_prev") > rules.min_hours,
                pl.col("start_shared") == shared,
            )
            .select(
                "vessel_id",
                "track",
                start="start_timestamp",
                end="timestamp",
                transceiver="start_transceiver",
                end_transceiver="transceiver",
                hours="hours_to_prev",
                distance_km="km_to_prev",
                start_lon="start_lon",
                start_lat="start_lat",
                end_lon="lon",
                end_lat="lat",
                prior_positions="start_prior_positions",
            )
        )
    return pl.concat(found).sort("vessel_id", "end", "track")


def silences(
    segmented: pl.DataFrame, watermark, rules: GapRules = GapRules()
) -> pl.DataFrame:
    """Tracks silent for over ``min_hours`` at the watermark: gaps that
    have begun but not ended.

    Args:
        segmented: ``segment.segment`` output.
        watermark: Time up to which every position has been seen.
        rules: Thresholds.

    Returns:
        pl.DataFrame: ``event_id`` (as ``gap_events`` will give it once
        the gap ends), ``vessel_id``, ``track``, ``start``, ``lon`` and
        ``lat``. A shared-ID segment silent for over ``shared_hours`` is
        left out, since the segmenter has closed it and no gap can end.
    """
    rows = gap_rows(segmented)
    rows = rows.with_columns(_shared=_shared(rows, rules.shared_hours))
    quiet = (pl.lit(watermark) - pl.col("timestamp")).dt.total_microseconds()
    by_vessel = (
        rows.group_by("vessel_id", maintain_order=True)
        .last()
        .filter(pl.col("_shared").not_())
        .with_columns(track=pl.col("vessel_id"))
    )
    by_segment = (
        rows.filter(pl.col("segment").is_not_null())
        .group_by("segment", maintain_order=True)
        .last()
        .filter("_shared", quiet <= rules.shared_hours * MICROS_PER_HOUR)
        .with_columns(track=pl.col("segment"))
    )
    columns = ["vessel_id", "track", "timestamp", "lon", "lat"]
    return (
        pl.concat([by_vessel.select(columns), by_segment.select(columns)])
        .filter(quiet > rules.min_hours * MICROS_PER_HOUR)
        .select(
            event_id=pl.concat_str(
                "track",
                pl.lit(DETECTOR),
                pl.col("timestamp").dt.strftime("%Y%m%dT%H%M%S"),
                separator=":",
            ),
            vessel_id="vessel_id",
            track="track",
            start="timestamp",
            lon="lon",
            lat="lat",
        )
    )


def _cells(lon: pl.Series, lat: pl.Series, resolution: int) -> pl.Series:
    """H3 cell of each position.

    Args:
        lon: Longitudes.
        lat: Latitudes.
        resolution: H3 resolution.

    Returns:
        pl.Series: Cell IDs as text.
    """
    return pl.Series(
        [h3.latlng_to_cell(y, x, resolution) for x, y in zip(lon, lat)],
        dtype=pl.String,
    )


def gap_events(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    segmented: pl.DataFrame,
    grid: pl.DataFrame,
    ports: PortDistance,
    heard: pl.DataFrame,
    rules: GapRules = GapRules(),
    mooring: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Find gaps and mark which are unexplained inside coverage.

    Args:
        segmented: ``segment.segment`` output.
        grid: ``coverage.coverage`` output (``h3_cell``, ``transceiver``,
            ``covered``); its cells set the H3 resolution.
        ports: Distance to the nearest anchorage; its projection is also
            used for the mooring radius.
        heard: ``h3_cell`` and ``day`` (UTC date) for every cell and day
            the reception grid heard any traffic.
        rules: Thresholds.
        mooring: ``Moorings.flags`` output for these gaps, kept by a caller
            that sees the positions in pieces; None to build it here from
            ``segmented`` alone.

    Returns:
        pl.DataFrame: One row per gap in ``GAP_SCHEMA``, ordered by vessel
        and start. ``event_id`` is the track and start time.
    """
    found = gap_candidates(segmented, rules)
    if found.is_empty():
        return pl.DataFrame(schema=GAP_SCHEMA)
    if mooring is None:
        mooring = Moorings(ports.projection).flags(
            gap_rows(segmented), found, rules.mooring_km
        )
    resolution = h3.get_resolution(grid["h3_cell"][0])
    found = found.join(mooring, on=CANDIDATE_KEY, how="left", maintain_order="left")
    found = found.with_columns(
        start_cell=_cells(found["start_lon"], found["start_lat"], resolution),
        end_cell=_cells(found["end_lon"], found["end_lat"], resolution),
        start_port_km=pl.Series(ports.km(found["start_lon"], found["start_lat"])),
        end_port_km=pl.Series(ports.km(found["end_lon"], found["end_lat"])),
        implied_kn=pl.col("distance_km") / 1.852 / pl.col("hours"),
    )
    found = found.with_columns(
        covered=_covered(found, grid, "start_cell", "transceiver")
        & _covered(found, grid, "end_cell", "end_transceiver"),
        away_from_port=(pl.col("start_port_km") >= rules.min_port_km)
        & (pl.col("end_port_km") >= rules.min_port_km),
        dense_before=pl.col("prior_positions") >= rules.min_prior_positions,
        cells_heard=_heard_throughout(found, heard),
    )
    return (
        found.with_columns(
            unexplained=pl.col("covered")
            & pl.col("away_from_port")
            & pl.col("away_from_mooring")
            & pl.col("dense_before")
            & pl.col("cells_heard"),
            event_id=pl.concat_str(
                "track",
                pl.lit(DETECTOR),
                pl.col("start").dt.strftime("%Y%m%dT%H%M%S"),
                separator=":",
            ),
        )
        .with_columns(
            long=pl.col("unexplained") & (pl.col("hours") >= rules.long_hours)
        )
        .select(pl.col(name).cast(dtype) for name, dtype in GAP_SCHEMA.items())
        .sort("vessel_id", "start", "track")
    )


def _covered(
    found: pl.DataFrame, grid: pl.DataFrame, cell: str, transceiver: str
) -> pl.Series:
    """Whether each gap end lies in a cell covered for its AIS class.

    Args:
        found: Gaps with the cell and class columns.
        grid: The coverage grid.
        cell: Column holding the end's cell.
        transceiver: Column holding the end's class.

    Returns:
        pl.Series: True where covered; a cell missing from the grid is not.
    """
    return (
        found.select(cell, transceiver)
        .with_row_index("_gap")
        .join(
            grid.select("h3_cell", "transceiver", "covered"),
            left_on=[cell, transceiver],
            right_on=["h3_cell", "transceiver"],
            how="left",
        )
        .sort("_gap")["covered"]
        .fill_null(False)
    )


def _heard_throughout(found: pl.DataFrame, heard: pl.DataFrame) -> pl.Series:
    """Whether both end cells heard traffic on every day a gap spans.

    Args:
        found: Gaps with ``start``, ``end``, ``start_cell`` and
            ``end_cell``.
        heard: ``h3_cell`` and ``day`` pairs with any traffic.

    Returns:
        pl.Series: True where every (end cell, day) pair was heard.
    """
    days = found.select(
        "start_cell",
        "end_cell",
        day=pl.date_ranges(pl.col("start").dt.date(), pl.col("end").dt.date()),
    ).with_row_index("_gap")
    needed = (
        days.unpivot(
            index=["_gap", "day"], on=["start_cell", "end_cell"], value_name="h3_cell"
        )
        .explode("day")
        .select("_gap", "h3_cell", "day")
    )
    marked = needed.join(
        heard.select("h3_cell", "day").unique().with_columns(_heard=pl.lit(True)),
        on=["h3_cell", "day"],
        how="left",
    )
    per_gap = marked.group_by("_gap").agg(ok=pl.col("_heard").fill_null(False).all())
    return days.select("_gap").join(per_gap, on="_gap", how="left").sort("_gap")["ok"]
