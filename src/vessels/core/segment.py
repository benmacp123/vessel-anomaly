"""Split each vessel's positions into segments, one per physical track.

Adapted from GFW's gpsdio-segment v3.0.0 (Apache-2.0, Copyright 2015-2016
SkyTruth), the segmenter pipe-segment runs; constants are
literature-sourced (GFW). Several transmitters can share one MMSI, so a
vessel keeps several open segments. Each position is compared with where
every open segment should be by now, dead-reckoned from its last speed and
course in both directions, and joins the segment it fits best. A position
that fits none starts a segment; one that fits two or more about equally
well closes them all and starts a new one, since it cannot be assigned
safely.

Differences from gpsdio-segment, all simplifications for this data:

- no five-point lookback, which drops earlier points when a later one
  fits better: an outlier here becomes a short segment of its own;
- no type 27 (long-range) handling: MarineCadastre has none;
- no daily fragments stitched into segments: a block is segmented in one
  pass;
- in the ambiguity check, each match's short-segment weight comes from
  its own segment. gpsdio-segment pairs matches with weights from every
  open segment, so the pairs shift when a segment does not match.

Rows are kept: positions without a usable speed (or moving with no course)
are ``bad`` and repeats of a moving position are ``repeat``, as GFW
classes them; neither gets a segment.
"""

import math
from dataclasses import dataclass, field

import polars as pl

# Literature-sourced (GFW gpsdio-segment v3.0.0 Matcher defaults; pipe-segment
# asserts MAX_HOURS is 24).
MAX_HOURS = 24.0
MAX_KNOTS = 25.0
PENALTY_HOURS = 4.0
HOURS_EXP = 0.5
BUFFER_HOURS = 0.25
PENALTY_SPEED = 5.0
SHAPE_FACTOR = 4.0
AMBIGUITY_FACTOR = 10.0
SHORT_SEG_THRESHOLD = 10
TRANSPONDER_MISMATCH_WEIGHT = 0.1
VERY_SLOW = 0.35
MAX_OPEN_SEGMENTS = 100
NM_PER_DEG_LAT = 60.0


@dataclass
class _Point:
    """One position, as the matcher reads it.

    Attributes:
        hours: Time since an arbitrary origin, hours.
        lon: Longitude, degrees.
        lat: Latitude, degrees.
        sog: Speed over ground, knots.
        cog: Course over ground, degrees; None when unknown and very slow.
        transceiver: AIS class.
    """

    hours: float
    lon: float
    lat: float
    sog: float
    cog: float | None
    transceiver: str | None


@dataclass
class _Segment:
    """An open segment: its last point, size and transceiver classes.

    Attributes:
        label: Segment ID written to the output.
        last: Most recent point.
        count: Points so far.
        transceivers: Classes seen on its last point.
    """

    label: str
    last: _Point
    count: int = 1
    transceivers: set = field(default_factory=set)


def _penalized_hours(hours: float) -> float:
    """Shorten long intervals: vessels rarely go straight for hours.

    Args:
        hours: Time between two points.

    Returns:
        float: The effective hours used for dead reckoning.
    """
    return hours / (1 + (max(hours, 0) / PENALTY_HOURS) ** (1 - HOURS_EXP))


def _expected(point: _Point, hours: float) -> tuple[float, float]:
    """Dead-reckon a point forward (or backward, for negative hours).

    Args:
        point: Start point.
        hours: Time to travel.

    Returns:
        tuple[float, float]: Expected longitude and latitude.
    """
    if point.cog is None:
        return point.lon, point.lat
    dist = point.sog * hours
    course = math.radians(90.0 - point.cog)
    deg_lat_per_nm = 1.0 / NM_PER_DEG_LAT
    deg_lon_per_nm = deg_lat_per_nm / (math.cos(math.radians(point.lat)) + 1e-3)
    return (
        point.lon + math.cos(course) * dist * deg_lon_per_nm,
        point.lat + math.sin(course) * dist * deg_lat_per_nm,
    )


def _discrepancy(p1: _Point, p2: _Point, hours: float) -> float:
    """How far two points are from being one track, in nautical miles.

    The smallest of GFW's three measures: the mean miss of each point
    dead-reckoned to the other; the distance apart if the vessel stayed
    put (times the shape factor); and the mean sideways offset from each
    point's course line, when the other point lies ahead within reach.

    Args:
        p1: Earlier point.
        p2: Later point.
        hours: Penalised hours between them.

    Returns:
        float: The discrepancy.
    """

    def wrap(x):
        """Wrap a longitude difference into [-180, 180)."""
        return (x + 180) % 360 - 180

    x2p, y2p = _expected(p1, hours)
    x1p, y1p = _expected(p2, -hours)
    nm_per_deg_lon = NM_PER_DEG_LAT * math.cos(math.radians(0.5 * (p1.lat + p2.lat)))
    d1 = 0.5 * (
        math.hypot(nm_per_deg_lon * wrap(x1p - p1.lon), NM_PER_DEG_LAT * (y1p - p1.lat))
        + math.hypot(
            nm_per_deg_lon * wrap(x2p - p2.lon), NM_PER_DEG_LAT * (y2p - p2.lat)
        )
    )
    dx = nm_per_deg_lon * wrap(p2.lon - p1.lon)
    dy = NM_PER_DEG_LAT * (p2.lat - p1.lat)
    dist = math.hypot(dx, dy)
    d2 = dist * SHAPE_FACTOR
    bearing = math.atan2(dy, dx)
    normals = []
    for point in (p1, p2):
        delta = math.radians(90 - (point.cog or 0.0)) - bearing
        tangential = math.cos(delta) * dist
        if 0 < tangential <= point.sog * hours:
            normals.append(abs(math.sin(delta)) * dist)
        else:
            normals.append(math.inf)
    d3 = 0.5 * sum(normals) * SHAPE_FACTOR
    return min(d1, d2, d3)


def _metric(seg: _Segment, point: _Point) -> float:
    """How well a point continues a segment; 0 means it cannot.

    Args:
        seg: An open segment.
        point: The next position.

    Returns:
        float: The match metric, higher is better.
    """
    hours = point.hours - seg.last.hours
    if hours > MAX_HOURS:
        return 0.0
    discrepancy = _discrepancy(seg.last, point, _penalized_hours(hours))
    padded = math.hypot(hours, BUFFER_HOURS)
    allowed = padded * MAX_KNOTS
    if discrepancy > allowed:
        return 0.0
    alpha = (MAX_KNOTS / PENALTY_SPEED) * discrepancy / allowed
    metric = math.exp(-(alpha**2)) / padded
    if point.transceiver not in seg.transceivers:
        metric *= TRANSPONDER_MISMATCH_WEIGHT
    return metric


class Segmenter:
    """Assigns positions to segments one at a time, so a stream can resume
    where it stopped.

    Positions must arrive in time order for each vessel. The clock is
    absolute (hours since the epoch), so the result does not depend on
    where a run starts. ``state`` holds each vessel's open segments and
    the last label stem used; ``to_dict`` and ``from_dict`` save and
    restore it as JSON.

    Attributes:
        state: Per vessel: ``open`` (label to ``_Segment``), and ``stem`` and
            ``n``, the last label's timestamp text and counter.
    """

    def __init__(self, state: dict | None = None):
        """Start empty or resume from ``state``.

        Args:
            state: Live state from another ``Segmenter``, or None.
        """
        self.state = state if state is not None else {}

    def to_dict(self) -> dict:
        """JSON-ready state.

        Returns:
            dict: Per vessel, open segments as plain dicts.
        """
        return {
            vessel: {
                "stem": record["stem"],
                "n": record["n"],
                "open": [
                    {
                        "label": seg.label,
                        "last": vars(seg.last),
                        "count": seg.count,
                        "transceivers": sorted(seg.transceivers, key=str),
                    }
                    for seg in record["open"].values()
                ],
            }
            for vessel, record in self.state.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Segmenter":
        """Restore a segmenter saved with ``to_dict``.

        Args:
            data: ``to_dict`` output.

        Returns:
            Segmenter: The equivalent segmenter.
        """
        state = {}
        for vessel, record in data.items():
            open_segments = {}
            for seg in record["open"]:
                open_segments[seg["label"]] = _Segment(
                    seg["label"],
                    _Point(**seg["last"]),
                    seg["count"],
                    set(seg["transceivers"]),
                )
            state[vessel] = {
                "stem": record["stem"],
                "n": record["n"],
                "open": open_segments,
            }
        return cls(state)

    def assign(self, vessel: str, point: _Point, stem: str) -> str:
        """Add one position and return its segment label.

        Args:
            vessel: The vessel ID, for segment labels.
            point: Its next position, in time order.
            stem: The position's timestamp as text, for a new label.

        Returns:
            str: The segment label.
        """
        record = self.state.setdefault(vessel, {"open": {}, "stem": None, "n": 0})
        open_segments = record["open"]
        for label in [
            s.label
            for s in open_segments.values()
            if point.hours - s.last.hours > MAX_HOURS
        ]:
            del open_segments[label]
        scored = [
            (m, s) for s in open_segments.values() if (m := _metric(s, point)) > 0
        ]
        if not scored:
            return self._start(vessel, record, point, stem)
        if len(scored) > 1:
            # Short segments are down-weighted so that a long track wins a
            # near tie with a fragment.
            weighted = []
            for metric, seg in scored:
                a = seg.count / SHORT_SEG_THRESHOLD
                weighted.append((metric * a / math.sqrt(1 + a**2), seg))
            weighted.sort(key=lambda pair: pair[0], reverse=True)
            best_metric = weighted[0][0]
            close = [s for m, s in weighted if m * AMBIGUITY_FACTOR >= best_metric]
            if len(close) > 1:
                for seg in close:
                    del open_segments[seg.label]
                return self._start(vessel, record, point, stem)
            best = weighted[0][1]
        else:
            best = scored[0][1]
        best.last = point
        best.count += 1
        best.transceivers = {point.transceiver}
        return best.label

    @staticmethod
    def _start(vessel: str, record: dict, point: _Point, stem: str) -> str:
        """Open a segment at a point, closing the stalest if too many are
        open.

        Labels are the vessel, the timestamp text and a counter for
        segments started in the same second; positions come in time order,
        so only the last stem can repeat.

        Args:
            vessel: Vessel ID.
            record: The vessel's state, updated in place.
            point: The first point.
            stem: Its timestamp as text.

        Returns:
            str: The new label.
        """
        open_segments = record["open"]
        while len(open_segments) >= MAX_OPEN_SEGMENTS:
            stalest = min(open_segments.values(), key=lambda s: s.last.hours)
            del open_segments[stalest.label]
        record["n"] = record["n"] + 1 if record["stem"] == stem else 1
        record["stem"] = stem
        label = f"{vessel}-{stem}-{record['n']}"
        open_segments[label] = _Segment(label, point, 1, {point.transceiver})
        return label


def point_status(frame: pl.DataFrame, seen: dict | None = None) -> pl.Series:
    """Class each row as a usable position, a bad one or a moving repeat.

    Args:
        frame: Cleaned positions sorted by vessel and time.
        seen: Per vessel, the keys of rows already classed, updated in
            place; None starts empty. A stream passes the same dict to
            every call.

    Returns:
        pl.Series: ``"position"``, ``"bad"`` or ``"repeat"`` per row.
    """
    seen = {} if seen is None else seen
    bad = (
        pl.col("sog").is_null()
        | (pl.col("sog").gt(VERY_SLOW) & pl.col("cog").is_null())
        | pl.col("lon").is_null()
        | pl.col("lat").is_null()
    )
    parts = [
        (pl.col("lat") * 60000).round(),
        (pl.col("lon") * 60000).round(),
        (pl.col("cog") * 10).round(),
        (pl.col("sog") * 10).round(),
        pl.col("heading"),
    ]
    key = pl.concat_str(
        [p.cast(pl.String).fill_null("-") for p in parts], separator="|"
    )
    marks = frame.select(
        "vessel_id", key=key, bad=bad, moving=pl.col("sog").gt(0).fill_null(False)
    )
    status = []
    # GFW skips a moving position seen before for the same vessel: the
    # same message relayed again, not a vessel that has not moved.
    for vessel, k, is_bad, moving in marks.iter_rows():
        keys = seen.setdefault(vessel, set())
        repeat = k in keys
        keys.add(k)
        status.append(
            "bad" if is_bad else "repeat" if repeat and moving else "position"
        )
    return pl.Series("status", status, dtype=pl.String)


def segment(
    frame: pl.DataFrame,
    segmenter: Segmenter | None = None,
    seen: dict | None = None,
) -> pl.DataFrame:
    """Add ``point_status`` and ``segment`` columns to cleaned positions.

    Args:
        frame: ``clean`` output: one vessel's rows together, in time order,
            with ``vessel_id``, ``timestamp``, ``lon``, ``lat``, ``sog``,
            ``cog``, ``heading`` and ``transceiver``.
        segmenter: State to resume from, updated in place; None starts
            empty.
        seen: Repeat keys to resume from (``point_status``); None starts
            empty.

    Returns:
        pl.DataFrame: The same rows and order with ``point_status`` and
        ``segment`` (null unless the status is ``position``).
    """
    segmenter = Segmenter() if segmenter is None else segmenter
    frame = frame.with_columns(point_status=point_status(frame, seen))
    usable = frame.with_row_index("_row").filter(pl.col("point_status") == "position")
    hours = (usable["timestamp"].dt.epoch("us") / 3.6e9).to_list()
    stems = usable["timestamp"].dt.strftime("%Y%m%dT%H%M%S").to_list()
    labels: list[str | None] = [None] * frame.height
    for row, vessel, h, lon, lat, sog, cog, cls, stem in zip(
        usable["_row"].to_list(),
        usable["vessel_id"].to_list(),
        hours,
        usable["lon"].to_list(),
        usable["lat"].to_list(),
        usable["sog"].to_list(),
        usable["cog"].to_list(),
        usable["transceiver"].to_list(),
        stems,
    ):
        point = _Point(
            float(h), lon, lat, float(sog), None if cog is None else float(cog), cls
        )
        labels[row] = segmenter.assign(vessel, point, stem)
    return frame.with_columns(segment=pl.Series(labels, dtype=pl.String))


def segment_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """One row per segment: vessel, span and size.

    Args:
        frame: ``segment`` output.

    Returns:
        pl.DataFrame: ``vessel_id``, ``segment``, ``start``, ``end``,
        ``points`` and ``short`` (fewer than ``SHORT_SEG_THRESHOLD`` points),
        sorted by vessel and start.
    """
    return (
        frame.filter(pl.col("segment").is_not_null())
        .group_by("vessel_id", "segment")
        .agg(
            start=pl.col("timestamp").min(),
            end=pl.col("timestamp").max(),
            points=pl.len(),
        )
        .with_columns(short=pl.col("points") < SHORT_SEG_THRESHOLD)
        .sort("vessel_id", "start", "segment")
    )


def overlapping_vessels(summary: pl.DataFrame) -> pl.DataFrame:
    """Vessels with two segments open at the same time (likely shared IDs).

    Args:
        summary: ``segment_summary`` output.

    Returns:
        pl.DataFrame: ``vessel_id`` and ``overlaps``, the number of segments
        that start before an earlier-starting segment of the same vessel
        has ended.
    """
    return (
        summary.with_columns(
            prev_end=pl.col("end").cum_max().shift().over("vessel_id"),
        )
        .filter(pl.col("start") < pl.col("prev_end"))
        .group_by("vessel_id")
        .agg(overlaps=pl.len())
        .sort("vessel_id")
    )
