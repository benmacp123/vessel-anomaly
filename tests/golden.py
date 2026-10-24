"""Golden scenarios: hand-built tracks, one per behaviour, with the events
and incidents they must produce.

Run ``make golden`` to rewrite ``tests/golden/*.json`` after a deliberate
change; ``tests/test_golden.py`` fails on any other difference. Every
scenario includes a beacon vessel reporting hourly, so the stream's
watermark moves while the others are silent.
"""

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import h3
import polars as pl
from shapely.geometry import box

from synthetic import T0, Track, frame
from vessels.core.distance import PortDistance, ShoreDistance
from vessels.core.geo import LocalProjection
from vessels.core.geofence import Rule
from vessels.sim.arrival import SCENARIOS, arrivals
from vessels.stream.micro import Inputs, replay, run_batch

FOLDER = Path(__file__).resolve().parent / "golden"
PROJECTION = LocalProjection(-69.0, 42.0)
CLOSURE = Rule(
    "hma",
    "closure",
    ("mobile_bottom",),
    date(2018, 4, 9),
    None,
    box(-69, 42, -68.8, 42.2),
)


def beacon() -> Track:
    """A vessel heard hourly for three days that raises nothing.

    Returns:
        Track: The beacon.
    """
    return Track("beacon", -66.5, 41.2).leg(3 * 24 * 60, 3.0, step=60)


def closure_tow() -> list[Track]:
    """Towing in a closure, a brief exit that merges, a transit between
    tows and a tow resumed after three hours.

    Returns:
        list[Track]: The tracks.
    """
    tow = Track("tow", -68.95, 42.1).leg(90, 3.5).leg(60, 9.0).leg(60, 3.5, cog=270)
    tow.leg(50, 3.5, cog=90).silence(3, 3.5).leg(40, 3.5)
    return [tow]


def loiter_drift() -> list[Track]:
    """Drifting far offshore: a long slow spell, a short one, then a slow
    spell with a nine-hour silence inside it.

    Returns:
        list[Track]: The tracks.
    """
    drift = Track("drift", -67.5, 42.5, transceiver="B").leg(60, 9.0)
    drift.leg(10 * 60, 0.5).leg(90, 8.0).leg(5 * 60, 0.4).silence(9, 0.4).leg(120, 0.4)
    return [drift]


def dark_gap() -> list[Track]:
    """An eight-hour silence at sea and a 30-hour one, with a second
    transmitter on the same ID heard during the first day.

    Returns:
        list[Track]: The tracks.
    """
    dark = Track("dark", -68.0, 41.5).leg(12 * 60, 8.0, step=3).silence(8, 8.0)
    dark.leg(4 * 60, 8.0, step=3).silence(30, 8.0).leg(60, 8.0)
    shared = Track("dark", -66.5, 43.8, start=T0 + timedelta(hours=2))
    shared.leg(30 * 60, 6.0, step=10)
    return [dark, shared]


def mooring() -> list[Track]:
    """A silence at the vessel's own mooring, 4 km from shore.

    Returns:
        list[Track]: The tracks.
    """
    moored = Track("moored", -70.45, 42.4).leg(14 * 60, 0.0, step=6).silence(9)
    moored.leg(6 * 60, 0.0, step=6)
    return [moored]


def loiter_edges() -> list[Track]:
    """An hour that starts fast and ends slow, and a slow run longer than
    the stream's 48-hour gap look-back.

    Returns:
        list[Track]: The tracks.
    """
    # The fast spell starts on the hour, so that hour reads fast until its
    # slow remainder arrives.
    slowdown = Track("slowdown", -67.5, 42.5).leg(60, 9.0).leg(9 * 60, 0.4)
    slowdown.leg(10, 6.0).leg(50, 0.2).leg(4 * 60, 0.3).leg(120, 9.0)
    long_drift = Track("long", -67.0, 43.0).leg(90, 9.0, step=6)
    long_drift.leg(60 * 60, 0.3, step=6).leg(120, 9.0, step=6)
    return [slowdown, long_drift]


KM_PER_DEG_LAT = 111.195


def visit(name, host, start, stops, km=0.3, bearing=0.0) -> Track:
    """A vessel that waits 5 km from a stopped host, runs in at 10 kn or
    so to ``km`` from it, stops, and runs out again, once per stop.

    Args:
        name: Vessel ID.
        host: The host's ``(lon, lat)``.
        start: First report.
        stops: ``(minutes_stopped, minutes_away_after)`` per stop.
        km: Distance from the host while stopped.
        bearing: Direction from the host to the waiting spot, degrees.

    Returns:
        Track: The vessel.
    """
    rad = math.radians(bearing)
    lat = host[1] + 5 * math.cos(rad) / KM_PER_DEG_LAT
    lon = host[0] + 5 * math.sin(rad) / (
        KM_PER_DEG_LAT * math.cos(math.radians(host[1]))
    )
    run_kn = (5 - km) / 1.852 / 0.25
    track = Track(name, lon, lat, start=start).leg(10, 0.0, step=5)
    for stopped, away in stops:
        track.leg(15, run_kn, cog=(bearing + 180) % 360, step=1)
        track.leg(stopped, 0.0, step=5)
        track.leg(15, run_kn, cog=bearing, step=1).leg(away, 0.0, step=5)
    return track


def meeting() -> list[Track]:
    """A three-hour meeting far offshore, one across midnight (two raw
    runs merged) and a near miss at 600 m.

    Returns:
        list[Track]: The tracks.
    """
    host = (-67.0, 41.8)
    return [
        Track("host", *host).leg(40 * 60, 0.0, step=5),
        visit("guest", host, T0 + timedelta(hours=1), [(180, 60)]),
        visit(
            "night",
            host,
            T0 + timedelta(hours=21, minutes=50),
            [(240, 60)],
            bearing=120,
        ),
        visit("near", host, T0 + timedelta(hours=8), [(180, 60)], 0.6, bearing=240),
    ]


def encounter_edges() -> list[Track]:
    """A guest that goes silent beside the host and is heard again the same
    evening, one whose two stops three hours apart merge, and one whose
    evening stop merges with another after midnight.

    Returns:
        list[Track]: The tracks.
    """
    host = (-66.0, 42.8)
    silent = visit("silent", host, T0 + timedelta(hours=1), [(135, 0)])
    # Silent from 03:40 while still beside the host, heard again that
    # evening at its waiting spot: the hour dead-reckoned after its 03:40
    # report then vanishes, so the stream must not settle the meeting
    # before midnight.
    silent.rows = [
        r for r in silent.rows if r["timestamp"] <= T0 + timedelta(hours=3, minutes=40)
    ]
    silent.time = T0 + timedelta(hours=20)
    silent.leg(150, 0.0, step=5)
    return [
        Track("host2", *host).leg(30 * 60, 0.0, step=5),
        silent,
        visit(
            "twice", host, T0 + timedelta(hours=6), [(150, 150), (150, 60)], bearing=180
        ),
        # Ends at 21:30, so the merge window, not midnight, decides when
        # it settles.
        visit(
            "late", host, T0 + timedelta(hours=19), [(125, 150), (150, 60)], bearing=90
        ),
    ]


def pair(names, lon, lat, start, legs) -> list[Track]:
    """Two vessels 200 m apart sailing the same legs.

    Args:
        names: The two vessel IDs.
        lon: Start longitude of the first.
        lat: Start latitude of the first.
        start: First report.
        legs: ``(minutes, sog)`` per leg, heading east.

    Returns:
        list[Track]: The two tracks.
    """
    tracks = [
        Track(names[0], lon, lat, start=start),
        Track(names[1], lon, lat + 0.2 / KM_PER_DEG_LAT, start=start),
    ]
    for track, step in zip(tracks, (10, 7)):
        for minutes, sog in legs:
            track.leg(minutes, sog, step=step)
    return tracks


def encounter_long() -> list[Track]:
    """Encounters longer than the stream's look-back: a raft-up of 80
    hours, five days together that fail on speed although their last
    three days would pass, and a slow evening followed by a fast day,
    which passes until the fast part is merged in.

    Returns:
        list[Track]: The tracks.
    """
    host = (-67.5, 43.5)
    return [
        # Heard for two days after the others, so the stream trims the
        # five-day pair's tail before the flush.
        Track("host3", *host).leg(156 * 60, 0.0, step=10),
        visit("raft", host, T0 + timedelta(hours=1), [(80 * 60, 0)]),
        *pair(("fasta", "fastb"), -69.8, 40.6, T0, [(72 * 60, 3.0), (48 * 60, 0.5)]),
        *pair(
            ("turna", "turnb"),
            -66.5,
            40.4,
            T0 + timedelta(hours=19),
            [(5 * 60, 0.3), (30 * 60, 3.0)],
        ),
    ]


GOLDEN = {
    "closure_tow": closure_tow,
    "loiter_drift": loiter_drift,
    "dark_gap": dark_gap,
    "mooring": mooring,
    "loiter_edges": loiter_edges,
    "meeting": meeting,
    "encounter_edges": encounter_edges,
    "encounter_long": encounter_long,
}


def positions(*names: str) -> pl.DataFrame:
    """The named scenarios' positions together, with the beacon.

    Args:
        *names: Scenario names from ``GOLDEN``.

    Returns:
        pl.DataFrame: Their positions.
    """
    tracks = [t for name in names for t in GOLDEN[name]()]
    return frame(*tracks, beacon())


def inputs_for(data: pl.DataFrame) -> Inputs:
    """Reference layers in which every cell the data touches is covered and
    heard on every day.

    Args:
        data: Positions.

    Returns:
        Inputs: Layers and default thresholds.
    """
    cells = sorted(
        {h3.latlng_to_cell(y, x, 6) for x, y in zip(data["lon"], data["lat"])}
    )
    grid = pl.DataFrame({"h3_cell": cells}).join(
        pl.DataFrame({"transceiver": ["A", "B"], "covered": [True, True]}),
        how="cross",
    )
    days = pl.date_range(T0.date(), T0.date() + timedelta(days=4), eager=True)
    heard = pl.DataFrame({"h3_cell": cells}).join(
        pl.DataFrame({"day": days}), how="cross"
    )
    return Inputs(
        closures=[CLOSURE],
        grid=grid,
        ports=PortDistance([-71.0], [41.0], PROJECTION),
        heard=heard,
        shore=ShoreDistance(box(-72, 40, -70.5, 44), PROJECTION),
    )


def _plain(value):
    """A JSON-ready value: times as text, floats rounded.

    Args:
        value: Any cell value.

    Returns:
        The value as JSON can hold it.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 6)
    return value


def build(name: str) -> dict:
    """Run one scenario through the batch detectors and an hourly stream.

    Args:
        name: Scenario name.

    Returns:
        dict: ``events`` by detector and ``incidents``, the stream's
        transitions, JSON-ready.
    """
    data = positions(name)
    inputs = inputs_for(data)
    events = {
        detector: [
            {k: _plain(v) for k, v in row.items()}
            for row in found.iter_rows(named=True)
        ]
        for detector, found in run_batch(data, inputs).items()
    }
    stream = replay(
        arrivals(data, SCENARIOS["ideal"]), inputs, timedelta(0), timedelta(hours=1)
    )
    keep = ("incident_id", "status", "resolution", "opened_at", "raised_at")
    incidents = [
        {k: _plain(v) for k, v in i.to_dict().items() if k in keep}
        for i in stream.incidents
    ]
    return {"events": events, "incidents": incidents}


def main():
    """Rewrite every golden file."""
    FOLDER.mkdir(exist_ok=True)
    for name in GOLDEN:
        path = FOLDER / f"{name}.json"
        path.write_text(json.dumps(build(name), indent=1) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
