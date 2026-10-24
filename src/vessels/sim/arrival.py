"""Seeded arrival scenarios: when each AIS position would reach the stream.

Adapted from collar-telemetry @ v0.1.0 (``telemetry/sim/arrival.py``).
Collars upload in batches; AIS positions are relayed one message at a time
by shore receivers, so the collar's upload schedule and coverage outages
are replaced by a per-message delay:

- **Delay**: each position arrives a uniform random time between zero and
  ``max_delay`` after it was recorded, so positions of one vessel can
  arrive out of order.
- **Duplicates** (``duplicate_share``): a position is also heard by a
  second receiver and arrives again with its own delay, as when receivers'
  ranges overlap.

Each scenario gives the stream's reorder ``hold``, longer than
``max_delay``, so no position arrives after the stream has moved past it
and the stream's output must equal the batch output. A scenario with a
hold shorter than its delay shows what late positions cost.

Randomness is seeded by the scenario seed and the vessel ID, so a vessel's
arrivals do not change when other vessels are added. All numbers are
provisional: MarineCadastre publishes archived positions with no arrival
times, so nothing here is measured.
"""

import hashlib
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ArrivalScenario:
    """One arrival pattern.

    Attributes:
        name: Short name used in results.
        max_delay: Longest delay between recording and arrival.
        duplicate_share: Chance that a position arrives twice.
        hold: Stream reorder hold suited to this scenario.
    """

    name: str
    max_delay: timedelta
    duplicate_share: float = 0.0
    hold: timedelta = timedelta(0)

    def __post_init__(self):
        """Reject impossible shares.

        Raises:
            ValueError: If ``duplicate_share`` is outside [0, 1).
        """
        if not 0.0 <= self.duplicate_share < 1.0:
            raise ValueError("duplicate_share must be in [0, 1)")


SCENARIOS: dict[str, ArrivalScenario] = {
    scenario.name: scenario
    for scenario in (
        # Baseline: every position arrives as it is recorded.
        ArrivalScenario("ideal", timedelta(0)),
        # Terrestrial network: up to a minute of relay delay, with 5% of
        # positions heard by two receivers.
        ArrivalScenario(
            "terrestrial", timedelta(minutes=1), 0.05, timedelta(minutes=2)
        ),
        # Receiver backhaul in batches: up to 20 minutes late.
        ArrivalScenario("backhaul", timedelta(minutes=20), 0.01, timedelta(minutes=30)),
        # As backhaul, with a hold too short: some positions arrive late.
        ArrivalScenario(
            "short_hold", timedelta(minutes=20), 0.01, timedelta(minutes=5)
        ),
    )
}


def _rng(seed: int, vessel: str) -> np.random.Generator:
    """A random source for one vessel under one seed.

    Args:
        seed: Scenario seed.
        vessel: Vessel ID.

    Returns:
        np.random.Generator: Independent of other vessels.
    """
    digest = hashlib.sha256(f"{seed}:{vessel}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def arrivals(
    frame: pl.DataFrame, scenario: ArrivalScenario, seed: int = 0
) -> pl.DataFrame:
    """Give every position an arrival time, adding duplicates.

    Args:
        frame: Positions with ``vessel_id`` and ``timestamp``.
        scenario: The arrival pattern.
        seed: Seed for the delays and duplicates.

    Returns:
        pl.DataFrame: ``frame``'s columns plus ``arrival``, with any
        duplicates, sorted by arrival time (ties by vessel and time).
    """
    max_us = scenario.max_delay // timedelta(microseconds=1)
    parts = []
    for (vessel,), part in frame.group_by("vessel_id", maintain_order=True):
        rng = _rng(seed, vessel)
        copies = rng.random(part.height) < scenario.duplicate_share
        both = pl.concat([part, part.filter(pl.Series(copies))])
        delay = rng.integers(0, max_us, both.height, endpoint=True)
        parts.append(
            both.with_columns(
                arrival=pl.col("timestamp") + pl.Series(delay).cast(pl.Duration("us"))
            )
        )
    if not parts:
        return frame.with_columns(arrival=pl.col("timestamp"))
    return pl.concat(parts).sort(
        "arrival", "vessel_id", "timestamp", maintain_order=True
    )
