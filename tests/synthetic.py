"""Synthetic AIS tracks built leg by leg, for detector tests and golden
scenarios. No real vessel's track is used anywhere in the tests."""

import math
from datetime import datetime, timedelta, timezone

import polars as pl

from vessels.io.marinecadastre import EXTRACT_SCHEMA

T0 = datetime(2024, 5, 6, 0, 0, tzinfo=timezone.utc)


class Track:
    """One vessel's positions, extended a leg at a time from where the last
    leg ended.

    Attributes:
        vessel: Vessel ID.
        lon: Current longitude.
        lat: Current latitude.
        time: Time of the last position written.
        transceiver: AIS class.
        rows: Positions so far.
    """

    def __init__(self, vessel, lon, lat, start=T0, transceiver="A"):
        """Place the vessel; nothing is written until the first leg.

        Args:
            vessel: Vessel ID.
            lon: Start longitude.
            lat: Start latitude.
            start: Time of the first position.
            transceiver: AIS class.
        """
        self.vessel = vessel
        self.lon = lon
        self.lat = lat
        self.time = start
        self.transceiver = transceiver
        self.rows: list[dict] = []

    def _write(self, sog, cog):
        """Append the current position.

        Args:
            sog: Reported speed, knots.
            cog: Reported course, degrees.
        """
        self.rows.append(
            {
                "vessel_id": self.vessel,
                "timestamp": self.time,
                "lon": self.lon,
                "lat": self.lat,
                "sog": sog,
                "cog": cog,
                "heading": None,
                "status": None,
                "vessel_type": 30,
                "length": 20.0,
                "transceiver": self.transceiver,
            }
        )

    def _move(self, minutes, sog, cog):
        """Dead-reckon the position forward.

        Args:
            minutes: Time elapsed.
            sog: Speed, knots.
            cog: Course, degrees.
        """
        nm = sog * minutes / 60
        self.lon += (
            nm * math.sin(math.radians(cog)) / (60 * math.cos(math.radians(self.lat)))
        )
        self.lat += nm * math.cos(math.radians(cog)) / 60
        self.time += timedelta(minutes=minutes)

    def leg(self, minutes, sog, cog=90.0, step=2.0):
        """Steam at constant speed, reporting every ``step`` minutes.

        The first leg starts with a position at the start; later legs start
        one step after the previous leg's last position.

        Args:
            minutes: Leg duration.
            sog: Speed, knots.
            cog: Course, degrees.
            step: Minutes between reports.

        Returns:
            Track: This track, for chaining.
        """
        if not self.rows:
            self._write(sog, cog)
        for _ in range(round(minutes / step)):
            self._move(step, sog, cog)
            self._write(sog, cog)
        return self

    def silence(self, hours, sog=0.0, cog=90.0):
        """Move on without reporting, as during an AIS gap.

        Args:
            hours: Gap length.
            sog: Speed during the gap, knots.
            cog: Course during the gap, degrees.

        Returns:
            Track: This track, for chaining.
        """
        self._move(hours * 60, sog, cog)
        return self


def frame(*tracks) -> pl.DataFrame:
    """The tracks' rows in the extract schema, sorted by vessel and time.

    Args:
        *tracks: ``Track`` objects.

    Returns:
        pl.DataFrame: Their positions.
    """
    rows = [row for t in tracks for row in t.rows]
    return (
        pl.DataFrame(rows, schema=EXTRACT_SCHEMA)
        .sort("vessel_id", "timestamp")
        .select(list(EXTRACT_SCHEMA))
    )
