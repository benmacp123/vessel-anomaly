"""Plant synthetic events in a block of real positions, with fixed seeds.

There is no US truth for any detector, so recall is measured on planted
events (PLAN.md, evaluation item 2). Each planter takes real positions and
returns them with planted cases plus a truth table, one row per case,
``label`` ``positive`` or a named negative. Every case is built from real
tracks, so cadence, jitter and speed noise are real:

- **gaps:** positions deleted from a real vessel's track, in place.
  Both ends must lie at sea in a covered cell for the vessel's class, at
  least the gap rule's distance from an anchorage, with the track
  continuous around the window, so the planted silence is the only one
  there. These are the population the gap claim covers (gaps at sea inside
  coverage); the rest of the rule (positions before, cells heard, own
  mooring, segmentation) is what recall measures. Lengths are drawn from
  real gaps over 6 hours (the caller passes them). Negatives are shorter
  than the rule's 6 hours.
- **closures:** a real towing run (2.5-5.5 kn) moved rigidly into a closure,
  clear of its exemption areas, as a new vessel ID: moving it within its
  own track would add a jump the segmenter splits on. Negatives are real
  transit runs (7-12 kn) moved to cross a closure.
- **encounters:** a new partner vessel that keeps a real donor vessel's
  timestamps (its cadence and class) and follows a real slow host offshore
  at a fixed range, closing from 5 km over 30 minutes and leaving the same
  way. Negatives are GFW's near misses: 600 m for 2 hours and 400 m for
  1.5 hours.
- **loitering:** a real offshore stretch slowed about its first position
  to a mean under 2 kn, between two hours of the real track before and
  after (shifted to join), as a new vessel ID. Negatives are too short
  (5-7 hours) or not slow enough (2.5-3.5 kn).

Positions passed in must carry ``covered``, ``port_km`` and ``shore_nm``
(``evaluation.injection.annotate``), so this module needs no H3 or
reference files. Planted cases never share a host vessel within 12 hours,
so they cannot interact. Labels are planted in a shuffled order, so a
shortfall of candidates falls on every label alike. A planter that runs out of candidates plants
fewer cases; the truth table says how many.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import shapely
from shapely.geometry.base import BaseGeometry

from vessels.core.geo import KM_PER_NM, LocalProjection

US_PER_HOUR = 3_600_000_000
# Columns planted rows keep; annotations are dropped from them.
EXTRACT_COLUMNS = (
    "vessel_id",
    "timestamp",
    "lon",
    "lat",
    "sog",
    "cog",
    "heading",
    "status",
    "vessel_type",
    "length",
    "transceiver",
)
TRUTH_SCHEMA = {
    "truth_id": pl.String,
    "kind": pl.String,
    "label": pl.String,
    "vessel_id": pl.String,
    "partner_id": pl.String,
    "closure_id": pl.String,
    "transceiver": pl.String,
    "start": pl.Datetime("us", "UTC"),
    "end": pl.Datetime("us", "UTC"),
    "lon": pl.Float64,
    "lat": pl.Float64,
}
# Prefix of planted vessel IDs; real IDs are 16 hex digits.
PLANTED = "inj"


@dataclass(frozen=True)
class InjectionRules:  # pylint: disable=too-many-instance-attributes
    """The frozen generator's settings (all provisional design choices).

    Attributes:
        seed: Base seed; each kind derives its own from it.
        gaps: Planted gap positives.
        gap_negatives: Planted gaps shorter than the rule.
        gap_negative_hours: Range of negative gap lengths.
        min_port_km: Least distance of a planted gap end from an anchorage.
        continuous_hours: Longest real step allowed around a planted case.
        closures: Planted towing runs per scored closure.
        closure_transits: Planted transit crossings in all.
        tow_hours: Range of planted towing run lengths.
        encounters: Planted encounter positives.
        near_misses: Planted near misses of each kind.
        encounter_hours: Range of planted encounter lengths.
        encounter_km: Range of planted encounter ranges.
        near_miss: (km, hours) of each near-miss kind.
        host_port_km: Least distance of an encounter host from an anchorage.
        host_max_kn: Greatest median reported speed of an encounter host.
        host_shore_nm: Least distance of an encounter host from shore, so
            the meeting is at sea (the rule drops runs on land).
        loiters: Planted loitering positives.
        loiter_negatives: Planted negatives of each kind.
        loiter_hours: Range of planted loitering lengths.
        loiter_kn: Range of the fastest slowed clock hour's speed.
        loiter_short_hours: Range of too-short lengths.
        loiter_fast_kn: Range of the slowest clock hour's speed in a
            too-fast negative.
        loiter_shore_nm: Least distance from shore of a slowed stretch.
        loiter_host_kn: Least speed of every clock hour of the real
            stretch, so only the slowing can make it loiter.
        buffer_hours: Gap kept around each case on its host vessel; over
            the encounter merge window (4 hours) so cases cannot merge.
        tries: Random candidates tried per case before giving up.
    """

    seed: int = 20241024
    gaps: int = 150
    gap_negatives: int = 50
    gap_negative_hours: tuple[float, float] = (4.0, 5.5)
    min_port_km: float = 10.0
    continuous_hours: float = 2.0
    closures: int = 25
    closure_transits: int = 50
    tow_hours: tuple[float, float] = (1.0, 3.0)
    encounters: int = 100
    near_misses: int = 50
    encounter_hours: tuple[float, float] = (2.5, 5.0)
    encounter_km: tuple[float, float] = (0.1, 0.4)
    near_miss: tuple[tuple[float, float], ...] = ((0.6, 2.0), (0.4, 1.5))
    host_port_km: float = 15.0
    host_max_kn: float = 1.5
    host_shore_nm: float = 1.0
    loiters: int = 100
    loiter_negatives: int = 50
    loiter_hours: tuple[float, float] = (9.0, 16.0)
    loiter_kn: tuple[float, float] = (0.3, 1.5)
    loiter_short_hours: tuple[float, float] = (5.0, 7.0)
    loiter_fast_kn: tuple[float, float] = (2.5, 3.5)
    loiter_shore_nm: float = 25.0
    loiter_host_kn: float = 2.0
    buffer_hours: float = 12.0
    tries: int = 200


@dataclass
class _Tracks:
    """A block's positions split per vessel, with a record of used windows.

    Attributes:
        frame: Positions sorted by vessel and time.
        starts: Each vessel's first row in ``frame``.
        used: Each vessel's windows (start, end in microseconds) already
            given to a case, buffer included.
    """

    frame: pl.DataFrame
    starts: dict[str, tuple[int, int]] = field(default_factory=dict)
    used: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def build(cls, frame: pl.DataFrame) -> "_Tracks":
        """Sort positions and index each vessel's rows.

        Args:
            frame: Annotated positions.

        Returns:
            _Tracks: The index.
        """
        frame = frame.sort("vessel_id", "timestamp").with_columns(
            _us=pl.col("timestamp").dt.epoch("us")
        )
        bounds = (
            frame.with_row_index("_row")
            .group_by("vessel_id", maintain_order=True)
            .agg(first=pl.col("_row").min(), last=pl.col("_row").max())
        )
        starts = {
            v: (int(a), int(b) + 1)
            for v, a, b in bounds.select("vessel_id", "first", "last").iter_rows()
        }
        return cls(frame, starts)

    def free(self, vessel: str, start: int, end: int) -> bool:
        """Whether a window on a vessel overlaps no case planted so far.

        Args:
            vessel: Vessel ID.
            start: Window start, microseconds.
            end: Window end, microseconds.

        Returns:
            bool: True when free.
        """
        return all(end < a or start > b for a, b in self.used.get(vessel, []))

    def take(self, vessel: str, start: int, end: int, buffer_us: int) -> None:
        """Reserve a window, with a buffer each side.

        Args:
            vessel: Vessel ID.
            start: Window start, microseconds.
            end: Window end, microseconds.
            buffer_us: Buffer, microseconds.
        """
        self.used.setdefault(vessel, []).append((start - buffer_us, end + buffer_us))


def _rng(seed: int, kind: str) -> np.random.Generator:
    """A random source for one kind of case.

    Args:
        seed: Base seed.
        kind: Case kind.

    Returns:
        np.random.Generator: Independent of the other kinds.
    """
    digest = hashlib.sha256(f"{seed}:{kind}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _shuffled(plans: list, rng: np.random.Generator) -> list:
    """Plans in a random order.

    Args:
        plans: Case plans.
        rng: Random source.

    Returns:
        list: The same plans, shuffled.
    """
    return [plans[k] for k in rng.permutation(len(plans))]


def _time_weighted(
    us: np.ndarray, vessels: np.ndarray, rows: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Candidate rows in a random order weighted by time, not by rows.

    Drawing rows uniformly would favour densely reported stretches (a
    vessel reporting every 10 seconds has 30 times the rows of one
    reporting every 5 minutes), so planted gaps would follow more positions
    than real ones. Each row is weighted by the time to its vessel's next
    row, capped at an hour, and ordered by Efraimidis-Spirakis keys.

    Args:
        us: Times in microseconds, sorted by vessel then time.
        vessels: Vessel ID of each row.
        rows: Candidate rows.
        rng: Random source.

    Returns:
        np.ndarray: ``rows`` in weighted random order (zero weights last).
    """
    step = np.diff(us, append=us[-1]).astype(float)
    step[np.append(vessels[1:] != vessels[:-1], True)] = 0.0
    weight = np.minimum(step[rows], US_PER_HOUR) / US_PER_HOUR
    keys = np.full(rows.size, -np.inf)
    positive = weight > 0
    keys[positive] = np.log(rng.random(positive.sum())) / weight[positive]
    return rows[np.argsort(-keys, kind="stable")]


def _max_step(us: np.ndarray, first: int, last: int) -> float:
    """Longest step between consecutive rows ``first`` to ``last``, in hours.

    Args:
        us: Times in microseconds.
        first: First row.
        last: Last row (inclusive).

    Returns:
        float: Longest step; infinity when the window has under two rows.
    """
    if last <= first:
        return float("inf")
    return float(np.diff(us[first : last + 1]).max()) / US_PER_HOUR


def _truth(rows: list[dict]) -> pl.DataFrame:
    """A truth table from case dictionaries.

    Args:
        rows: One dictionary per case, keyed by ``TRUTH_SCHEMA`` names.

    Returns:
        pl.DataFrame: The truth table.
    """
    full = [{name: row.get(name) for name in TRUTH_SCHEMA} for row in rows]
    return pl.DataFrame(full, schema=TRUTH_SCHEMA)


def _extract(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep only the extract's columns.

    Args:
        frame: Positions with annotations.

    Returns:
        pl.DataFrame: Positions in the extract schema.
    """
    return frame.select(EXTRACT_COLUMNS)


def plant_gaps(
    frame: pl.DataFrame, lengths: np.ndarray, rules: InjectionRules = InjectionRules()
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Delete positions to plant gaps in real tracks.

    Args:
        frame: Annotated positions (``covered``, ``port_km``).
        lengths: Real gap lengths in hours to draw positives from.
        rules: Generator settings.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Positions with the planted gaps,
        in the extract schema, and the truth table.
    """
    rng = _rng(rules.seed, "gap")
    tracks = _Tracks.build(frame)
    data = tracks.frame
    us = data["_us"].to_numpy()
    ok = (data["covered"] & (data["port_km"] >= rules.min_port_km)).to_numpy()
    vessels = data["vessel_id"].to_numpy()
    plans = [("positive", h) for h in rng.choice(lengths, rules.gaps)] + [
        ("short", h)
        for h in rng.uniform(*rules.gap_negative_hours, rules.gap_negatives)
    ]
    plans = _shuffled(plans, rng)
    order = _time_weighted(us, vessels, np.flatnonzero(ok), rng)
    drop = np.zeros(data.height, dtype=bool)
    cases, cursor = [], 0
    buffer_us = int(rules.buffer_hours * US_PER_HOUR)
    for label, hours in plans:
        while cursor < order.size:
            i = int(order[cursor])
            cursor += 1
            first, last = tracks.starts[vessels[i]]
            end_us = us[i] + int(hours * US_PER_HOUR)
            j = int(np.searchsorted(us[first:last], end_us)) + first
            if j >= last or not ok[j] or j - i < 2:
                continue
            after = int(np.searchsorted(us[first:last], us[j] + US_PER_HOUR))
            # The track must be continuous across the window and an hour
            # past it, so the planted silence is the only one there and no
            # real gap starts where it ends. Reporting before the window is
            # left as it was: thin reporting before a gap is part of what
            # recall measures.
            if _max_step(us, i, min(first + after, last - 1)) > rules.continuous_hours:
                continue
            if not tracks.free(vessels[i], us[i], us[j]):
                continue
            tracks.take(vessels[i], us[i], us[j], buffer_us)
            drop[i + 1 : j] = True
            cases.append(
                {
                    "truth_id": f"gap-{len(cases):03d}",
                    "kind": "gap",
                    "label": label,
                    "vessel_id": vessels[i],
                    "transceiver": data["transceiver"][i],
                    "start": data["timestamp"][i],
                    "end": data["timestamp"][j],
                    "lon": data["lon"][i],
                    "lat": data["lat"][i],
                }
            )
            break
    return _extract(data.filter(pl.Series(~drop))), _truth(cases)


def _runs(
    frame: pl.DataFrame, speed: pl.Expr, hours: tuple[float, float], step_hours: float
) -> pl.DataFrame:
    """Maximal runs of rows where ``speed`` holds, with no long step.

    Args:
        frame: ``_Tracks.frame``.
        speed: Condition each row must meet.
        hours: Least and greatest run length kept (longer runs are kept
            and cut later).
        step_hours: Longest step inside a run.

    Returns:
        pl.DataFrame: ``vessel_id``, ``first`` and ``last`` row, ``hours``.
    """
    step = pl.col("_us").diff().over("vessel_id") / US_PER_HOUR
    marked = frame.with_row_index("_row").with_columns(
        _in=speed,
        _break=step.is_null() | (step > step_hours),
    )
    marked = marked.with_columns(
        _run=(pl.col("_in").not_() | pl.col("_break") | pl.col("_in").shift(1).not_())
        .fill_null(True)
        .cum_sum()
    )
    return (
        marked.filter("_in")
        .group_by("_run", maintain_order=True)
        .agg(
            vessel_id=pl.col("vessel_id").first(),
            first=pl.col("_row").min(),
            last=pl.col("_row").max(),
            hours=(pl.col("_us").max() - pl.col("_us").min()) / US_PER_HOUR,
        )
        .filter(pl.col("hours") >= hours[0])
        .drop("_run")
    )


def _cut(
    us: np.ndarray, first: int, last: int, hours: float, rng: np.random.Generator
) -> tuple[int, int]:
    """A random sub-run of about ``hours`` within rows ``first`` to ``last``.

    Args:
        us: Times in microseconds.
        first: First row of the run.
        last: Last row of the run.
        hours: Wanted length.
        rng: Random source.

    Returns:
        tuple[int, int]: First and last row of the sub-run, which may end
        early when the run is shorter than ``hours``.
    """
    span = int(hours * US_PER_HOUR)
    latest = int(np.searchsorted(us[first : last + 1], us[last] - span, "right"))
    a = first + int(rng.integers(0, max(latest, 1)))
    # The first row at or after the wanted end, so the sub-run covers the
    # whole span whenever the run is long enough.
    b = first + int(np.searchsorted(us[first : last + 1], us[a] + span))
    return a, min(max(b, a), last)


def _new_vessel(piece: pl.DataFrame, vessel: str) -> pl.DataFrame:
    """A copy of a track piece under a new vessel ID.

    Args:
        piece: Rows of one real vessel.
        vessel: New ID.

    Returns:
        pl.DataFrame: The piece in the extract schema.
    """
    return _extract(piece.with_columns(vessel_id=pl.lit(vessel)))


def _place(
    x: np.ndarray,
    y: np.ndarray,
    area: BaseGeometry,
    rng: np.random.Generator,
    whole: bool,
    tries: int = 200,
) -> tuple[float, float] | None:
    """An offset that moves a projected run into an area.

    Args:
        x: Run's x in kilometres.
        y: Run's y in kilometres.
        area: Projected polygon.
        rng: Random source.
        whole: True when every point must land inside; otherwise the run's
            middle point must, and some point outside (a crossing).
        tries: Random targets tried.

    Returns:
        tuple[float, float] | None: The offset, or None if none fits.
    """
    xmin, ymin, xmax, ymax = area.bounds
    mid = x.size // 2
    for _ in range(tries):
        tx, ty = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        if not shapely.contains_xy(area, tx, ty):
            continue
        dx, dy = tx - x[mid], ty - y[mid]
        inside = shapely.contains_xy(area, x + dx, y + dy)
        if inside.all() if whole else not inside.all():
            return float(dx), float(dy)
    return None


def plant_closures(
    frame: pl.DataFrame,
    areas: dict[str, BaseGeometry],
    projection: LocalProjection,
    rules: InjectionRules = InjectionRules(),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Move real towing runs into closures, and transits across them.

    Args:
        frame: Annotated positions.
        areas: Closure ID to polygon (longitude, latitude), exemption areas
            already removed.
        projection: Projection for moving runs.
        rules: Generator settings.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Positions plus the planted
        vessels, in the extract schema, and the truth table.
    """
    rng = _rng(rules.seed, "closure")
    tracks = _Tracks.build(frame)
    data = tracks.frame
    us = data["_us"].to_numpy()
    lon, lat = data["lon"].to_numpy(), data["lat"].to_numpy()
    projected = {k: projection.geometry(g) for k, g in areas.items()}
    tows = _runs(data, pl.col("sog").is_between(2.5, 5.5), rules.tow_hours, 10 / 60)
    transits = _runs(data, pl.col("sog").is_between(7.0, 12.0), (0.5, 2.0), 10 / 60)
    plans = [
        ("positive", closure)
        for closure in sorted(projected)
        for _ in range(rules.closures)
    ]
    plans += [
        ("transit", sorted(projected)[k % len(projected)])
        for k in range(rules.closure_transits)
    ]
    plans = _shuffled(plans, rng)
    pools = {
        "positive": rng.permutation(tows.height),
        "transit": rng.permutation(transits.height),
    }
    cursors = {"positive": 0, "transit": 0}
    pieces, cases = [], []
    for label, closure in plans:
        runs = tows if label == "positive" else transits
        pool = pools[label]
        while cursors[label] < pool.size:
            _, first, last, _ = runs.row(int(pool[cursors[label]]))
            cursors[label] += 1
            if label == "positive":
                first, last = _cut(us, first, last, rng.uniform(*rules.tow_hours), rng)
            if last - first < 14:
                continue
            x, y = projection.forward(lon[first : last + 1], lat[first : last + 1])
            offset = _place(x, y, projected[closure], rng, whole=label == "positive")
            if offset is None:
                continue
            new_lon, new_lat = projection.inverse(x + offset[0], y + offset[1])
            vessel_id = f"{PLANTED}-closure-{len(cases):03d}"
            piece = data.slice(first, last - first + 1).with_columns(
                lon=pl.Series(new_lon), lat=pl.Series(new_lat)
            )
            pieces.append(_new_vessel(piece, vessel_id))
            cases.append(
                {
                    "truth_id": f"closure-{len(cases):03d}",
                    "kind": "closure",
                    "label": label,
                    "vessel_id": vessel_id,
                    "closure_id": closure,
                    "transceiver": piece["transceiver"][0],
                    "start": piece["timestamp"][0],
                    "end": piece["timestamp"][-1],
                    "lon": float(new_lon[0]),
                    "lat": float(new_lat[0]),
                }
            )
            break
    return pl.concat([_extract(data), *pieces]), _truth(cases)


def _range_km(
    times: np.ndarray, start: int, end: int, near_km: float, far_km: float = 5.0
) -> np.ndarray:
    """Partner range at each time: closing, holding, then opening.

    Args:
        times: Partner times, microseconds.
        start: Meeting start, microseconds.
        end: Meeting end, microseconds.
        near_km: Range held during the meeting.
        far_km: Range before closing and after opening.

    Returns:
        np.ndarray: Range in kilometres.
    """
    ramp = 0.5 * US_PER_HOUR
    closing = np.clip((start - times) / ramp, 0.0, 1.0)
    opening = np.clip((times - end) / ramp, 0.0, 1.0)
    return near_km + (far_km - near_km) * np.maximum(closing, opening)


def plant_encounters(  # pylint: disable=too-many-locals
    frame: pl.DataFrame,
    projection: LocalProjection,
    rules: InjectionRules = InjectionRules(),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add partner vessels that meet real slow hosts offshore.

    Args:
        frame: Annotated positions (``port_km``, ``shore_nm``).
        projection: Projection for the partner's offsets.
        rules: Generator settings.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Positions plus the partners, in
        the extract schema, and the truth table.
    """
    rng = _rng(rules.seed, "encounter")
    tracks = _Tracks.build(frame)
    data = tracks.frame
    us = data["_us"].to_numpy()
    lon, lat = data["lon"].to_numpy(), data["lat"].to_numpy()
    slow = pl.col("sog").fill_null(99.0) <= 2 * rules.host_max_kn
    away = (pl.col("port_km") >= rules.host_port_km) & (
        pl.col("shore_nm") >= rules.host_shore_nm
    )
    hosts = _runs(data, slow & away, (rules.encounter_hours[0] + 2.5, 1e9), 0.5)
    donors = _runs(data, pl.lit(True), (rules.encounter_hours[1] + 2.5, 1e9), 0.5)
    plans = [
        ("positive", *p)
        for p in zip(
            rng.uniform(*rules.encounter_km, rules.encounters),
            rng.uniform(*rules.encounter_hours, rules.encounters),
        )
    ]
    for k, (km, hours) in enumerate(rules.near_miss):
        plans += [(f"near_miss_{k}", km, hours)] * rules.near_misses
    pieces, cases = [], []
    buffer_us = int(rules.buffer_hours * US_PER_HOUR)
    for label, km, hours in _shuffled(plans, rng):
        for _ in range(rules.tries if hosts.height else 0):
            host, first, last, _ = hosts.row(int(rng.integers(hosts.height)))
            # The meeting sits an hour inside the host's slow run, so the
            # partner's approach and departure track a real host position.
            a, b = _cut(us, first, last, hours + 2.5, rng)
            start = us[a] + US_PER_HOUR
            end = start + int(hours * US_PER_HOUR)
            if us[b] < end + US_PER_HOUR or not tracks.free(host, us[a], us[b]):
                continue
            if np.median(data["sog"].to_numpy()[a : b + 1]) > rules.host_max_kn:
                continue
            donor, d_first, d_last, _ = donors.row(int(rng.integers(donors.height)))
            if donor == host:
                continue
            shift = us[a] - us[d_first]
            # Repeated donor times would give the partner zero-length steps.
            times = np.unique(us[d_first : d_last + 1] + shift)
            times = times[times <= us[b]]
            x, y = projection.forward(lon[a : b + 1], lat[a : b + 1])
            hx, hy = np.interp(times, us[a : b + 1], x), np.interp(
                times, us[a : b + 1], y
            )
            bearing = rng.uniform(0, 2 * np.pi)
            reach = _range_km(times, start, end, km)
            noise = rng.normal(0.0, 0.01, (2, times.size))
            px = hx + reach * np.sin(bearing) + noise[0]
            py = hy + reach * np.cos(bearing) + noise[1]
            p_lon, p_lat = projection.inverse(px, py)
            step_h = np.diff(times, prepend=times[0] - 60_000_000) / US_PER_HOUR
            step_km = np.hypot(np.diff(px, prepend=px[0]), np.diff(py, prepend=py[0]))
            course = np.degrees(
                np.arctan2(np.diff(px, prepend=px[0]), np.diff(py, prepend=py[0]))
            )
            tracks.take(host, us[a], us[b], buffer_us)
            partner = f"{PLANTED}-encounter-{len(cases):03d}"
            donor_row = data.row(d_first, named=True)
            pieces.append(
                pl.DataFrame(
                    {
                        "vessel_id": partner,
                        "timestamp": pl.Series(times).cast(pl.Datetime("us", "UTC")),
                        "lon": p_lon,
                        "lat": p_lat,
                        "sog": (step_km / KM_PER_NM / step_h).astype(np.float32),
                        "cog": (course % 360).astype(np.float32),
                        "heading": None,
                        "status": None,
                        "vessel_type": donor_row["vessel_type"],
                        "length": donor_row["length"],
                        "transceiver": donor_row["transceiver"],
                    }
                ).cast({k: v for k, v in data.select(EXTRACT_COLUMNS).schema.items()})
            )
            cases.append(
                {
                    "truth_id": f"encounter-{len(cases):03d}",
                    "kind": "encounter",
                    "label": label,
                    "vessel_id": host,
                    "partner_id": partner,
                    "transceiver": data["transceiver"][a],
                    "start": pl.Series([start]).cast(pl.Datetime("us", "UTC"))[0],
                    "end": pl.Series([end]).cast(pl.Datetime("us", "UTC"))[0],
                    "lon": float(lon[a]),
                    "lat": float(lat[a]),
                }
            )
            break
    return pl.concat([_extract(data), *pieces]), _truth(cases)


def _hourly_kn(us: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Implied speed in each UTC clock hour, as the loitering rule buckets it.

    Each step counts in the hour its later position falls in.

    Args:
        us: Times in microseconds, ascending.
        x: Projected x in kilometres.
        y: Projected y in kilometres.

    Returns:
        np.ndarray: Knots per clock hour with step time; empty with under
        two positions.
    """
    if us.size < 2:
        return np.zeros(0)
    km = np.hypot(np.diff(x), np.diff(y))
    hours = np.diff(us) / US_PER_HOUR
    keys, index = np.unique(us[1:] // US_PER_HOUR, return_inverse=True)
    total_km = np.bincount(index, km, keys.size)
    total_h = np.bincount(index, hours, keys.size)
    keep = total_h > 0
    return total_km[keep] / KM_PER_NM / total_h[keep]


def plant_loitering(  # pylint: disable=too-many-locals
    frame: pl.DataFrame,
    projection: LocalProjection,
    rules: InjectionRules = InjectionRules(),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Slow real offshore stretches into loitering, as new vessels.

    Args:
        frame: Annotated positions (``shore_nm``).
        projection: Projection for scaling the stretch.
        rules: Generator settings.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Positions plus the planted
        vessels, in the extract schema, and the truth table.
    """
    rng = _rng(rules.seed, "loitering")
    tracks = _Tracks.build(frame)
    data = tracks.frame
    us = data["_us"].to_numpy()
    lon, lat = data["lon"].to_numpy(), data["lat"].to_numpy()
    offshore = pl.col("shore_nm") >= rules.loiter_shore_nm
    shortest = min(rules.loiter_hours[0], rules.loiter_short_hours[0]) + 4.0
    stretches = _runs(data, offshore, (shortest, 1e9), 0.5)
    n = rules.loiters
    plans = list(
        zip(
            ["positive"] * n,
            rng.uniform(*rules.loiter_hours, n),
            rng.uniform(*rules.loiter_kn, n),
        )
    )
    m = rules.loiter_negatives
    plans += zip(
        ["short"] * m,
        rng.uniform(*rules.loiter_short_hours, m),
        rng.uniform(*rules.loiter_kn, m),
    )
    plans += zip(
        ["fast"] * m,
        rng.uniform(*rules.loiter_hours, m),
        rng.uniform(*rules.loiter_fast_kn, m),
    )
    pieces, cases = [], []
    buffer_us = int(rules.buffer_hours * US_PER_HOUR)
    for label, hours, knots in _shuffled(plans, rng):
        for _ in range(rules.tries if stretches.height else 0):
            vessel, first, last, _ = stretches.row(int(rng.integers(stretches.height)))
            a, b = _cut(us, first, last, hours + 4.0, rng)
            start = us[a] + 2 * US_PER_HOUR
            end = start + int(hours * US_PER_HOUR)
            i = int(np.searchsorted(us[a : b + 1], start)) + a
            j = int(np.searchsorted(us[a : b + 1], end, "right")) - 1 + a
            if us[b] < end + 2 * US_PER_HOUR or j <= i:
                continue
            if not tracks.free(vessel, us[a], us[b]):
                continue
            x, y = projection.forward(lon[a : b + 1], lat[a : b + 1])
            si, sj = i - a, j - a
            if _hourly_kn(us[a : b + 1], x, y).min() < rules.loiter_host_kn:
                continue
            hourly = _hourly_kn(us[i : j + 1], x[si : sj + 1], y[si : sj + 1])
            # Positives and short negatives are slow in every hour; fast
            # negatives are not slow in any.
            scale = knots / (hourly.min() if label == "fast" else hourly.max())
            tracks.take(vessel, us[a], us[b], buffer_us)
            nx, ny = x.copy(), y.copy()
            nx[si : sj + 1] = x[si] + scale * (x[si : sj + 1] - x[si])
            ny[si : sj + 1] = y[si] + scale * (y[si : sj + 1] - y[si])
            # The two hours after rejoin where the slowed stretch ended.
            nx[sj + 1 :] += nx[sj] - x[sj]
            ny[sj + 1 :] += ny[sj] - y[sj]
            new_lon, new_lat = projection.inverse(nx, ny)
            sog = data["sog"].to_numpy()[a : b + 1].astype(float).copy()
            sog[si : sj + 1] *= scale
            vessel_id = f"{PLANTED}-loitering-{len(cases):03d}"
            piece = data.slice(a, b - a + 1).with_columns(
                lon=pl.Series(new_lon),
                lat=pl.Series(new_lat),
                sog=pl.Series(sog).cast(pl.Float32),
            )
            pieces.append(_new_vessel(piece, vessel_id))
            cases.append(
                {
                    "truth_id": f"loitering-{len(cases):03d}",
                    "kind": "loitering",
                    "label": label,
                    "vessel_id": vessel_id,
                    "transceiver": piece["transceiver"][0],
                    "start": data["timestamp"][i],
                    "end": data["timestamp"][j],
                    "lon": float(new_lon[si]),
                    "lat": float(new_lat[si]),
                }
            )
            break
    return pl.concat([_extract(data), *pieces]), _truth(cases)
