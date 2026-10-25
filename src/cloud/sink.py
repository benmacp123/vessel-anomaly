"""Alert sink: each tick's settled events as Parquet and its incident
transitions as JSON, plus one GeoJSON file of every committed event for a
map.

Layout::

    parts/<tick>-<detector>.parquet   events settled at that tick
    parts/<tick>-incidents.json       incident transitions at that tick
    alerts.geojson                    every committed event

A tick's parts are written before its checkpoint commits, so a crash
between the two cannot lose them; on restart, parts of ticks after the
committed one are removed and that work is redone. Readers that must not
see work that may be redone read up to the committed tick (``read``).

The GeoJSON is derived and rebuilt from committed parts. Its coordinates
are rounded to 0.01 degrees (about 1 km): a hashed vessel ID does not
anonymise an exact track, and a map needs no more.
"""

from pathlib import Path

import polars as pl

from cloud.files import PARTIAL, read_json, write_json, write_parquet
from vessels.core.lifecycle import Incident
from vessels.stream.micro import ENCOUNTER, GAP, ORDER, SCHEMAS

INCIDENTS = "incidents"
DIGITS = 2


def _name(tick: int) -> str:
    """A tick's file prefix, padded so names sort as numbers.

    Args:
        tick: Tick number.

    Returns:
        str: The prefix.
    """
    return f"{tick:08d}"


def _tick(path: Path) -> int:
    """The tick a part belongs to.

    Args:
        path: A part file.

    Returns:
        int: Tick number.
    """
    return int(path.name.split("-", 1)[0])


def _point(lon: float, lat: float) -> list[float]:
    """Rounded GeoJSON coordinates.

    Args:
        lon: Longitude.
        lat: Latitude.

    Returns:
        list[float]: ``[lon, lat]`` to ``DIGITS`` places.
    """
    return [round(lon, DIGITS), round(lat, DIGITS)]


def feature(detector: str, row: dict) -> dict:
    """A GeoJSON feature for one event.

    A gap is drawn as the jump from where the vessel went dark to where it
    reappeared; an encounter at the pair's mean position; anything else
    where it started.

    Args:
        detector: Detector name.
        row: The event's row.

    Returns:
        dict: A GeoJSON feature with the event's ID, vessel, times and
        hours.
    """
    if detector == GAP:
        geometry = {
            "type": "LineString",
            "coordinates": [
                _point(row["start_lon"], row["start_lat"]),
                _point(row["end_lon"], row["end_lat"]),
            ],
        }
    elif detector == ENCOUNTER:
        geometry = {
            "type": "Point",
            "coordinates": _point(row["mean_lon"], row["mean_lat"]),
        }
    else:
        geometry = {
            "type": "Point",
            "coordinates": _point(row["start_lon"], row["start_lat"]),
        }
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "detector": detector,
            "event_id": row["event_id"],
            "vessel_id": row["vessel_id"],
            "start": row["start"].isoformat(),
            "end": row["end"].isoformat(),
            "hours": round(row["hours"], 2),
        },
    }


class FileSink:
    """Alerts in a local directory.

    Attributes:
        root: The sink directory.
    """

    def __init__(self, root: Path):
        """Open a sink, creating its directory if absent.

        Args:
            root: The sink directory.
        """
        self.root = root
        (root / "parts").mkdir(parents=True, exist_ok=True)

    def write(
        self, tick: int, events: dict[str, pl.DataFrame], incidents: list[Incident]
    ) -> None:
        """Write one tick's output; writing a tick again replaces it.

        Args:
            tick: Tick number.
            events: Settled events by detector (``MicroBatchStream.drain``).
            incidents: Incident transitions.
        """
        self.discard_after(tick - 1)
        for detector, frame in events.items():
            if frame.height:
                write_parquet(
                    self.root / "parts" / f"{_name(tick)}-{detector}.parquet", frame
                )
        if incidents:
            write_json(
                self.root / "parts" / f"{_name(tick)}-{INCIDENTS}.json",
                [incident.to_dict() for incident in incidents],
            )

    def discard_after(self, tick: int) -> None:
        """Remove parts of ticks after one, whose work will be redone, and
        any part a crash left half written.

        Args:
            tick: The last tick to keep.
        """
        for path in (self.root / "parts").iterdir():
            if _tick(path) > tick or path.name.endswith(PARTIAL):
                path.unlink()

    def read(self, upto: int) -> tuple[dict[str, pl.DataFrame], list[Incident]]:
        """Every event and incident transition up to a tick.

        Args:
            upto: The last tick to read, normally the committed one.

        Returns:
            tuple: Events by detector in the batch detectors' order, and
            incident transitions in tick order.
        """
        parts = sorted(p for p in (self.root / "parts").iterdir() if _tick(p) <= upto)
        events = {}
        for detector, schema in SCHEMAS.items():
            frames = [
                pl.read_parquet(p) for p in parts if p.stem.endswith(f"-{detector}")
            ]
            events[detector] = pl.concat([pl.DataFrame(schema=schema)] + frames).sort(
                ORDER[detector]
            )
        incidents = [
            Incident.from_dict(row)
            for p in parts
            if p.name.endswith(f"-{INCIDENTS}.json")
            for row in read_json(p)
        ]
        return events, incidents

    def publish(self, upto: int) -> int:
        """Rewrite the GeoJSON from every event up to a tick.

        Args:
            upto: The committed tick.

        Returns:
            int: Features written.
        """
        events, _ = self.read(upto)
        features = [
            feature(detector, row)
            for detector, frame in events.items()
            for row in frame.iter_rows(named=True)
        ]
        write_json(
            self.root / "alerts.geojson",
            {"type": "FeatureCollection", "features": features},
        )
        return len(features)
