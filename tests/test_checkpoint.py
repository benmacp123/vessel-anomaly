"""Tests for stream checkpoints: stopping between batches, writing the state
out as Parquet and JSON, reading it back and continuing gives what an
uninterrupted stream gives."""

import io
import json
from datetime import timedelta

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from golden import inputs_for, positions
from synthetic import T0, Track, frame
from vessels.sim.arrival import SCENARIOS, arrivals
from vessels.stream.checkpoint import restore, snapshot
from vessels.stream.micro import ORDER, MicroBatchStream, batches, replay

DATA = positions("closure_tow", "loiter_drift", "dark_gap", "mooring", "meeting")
INPUTS = inputs_for(DATA)


def through_files(stream: MicroBatchStream) -> MicroBatchStream:
    """Save a stream as Parquet bytes and JSON text, and load it back.

    Args:
        stream: The stream to save.

    Returns:
        MicroBatchStream: The stream rebuilt from what was written.
    """
    frames, meta = snapshot(stream)
    read = {}
    for name, table in frames.items():
        buffer = io.BytesIO()
        table.write_parquet(buffer)
        buffer.seek(0)
        read[name] = pl.read_parquet(buffer)
    return restore(read, json.loads(json.dumps(meta)), INPUTS)


def stopped_run(arrived, hold, length, stops, drain):
    """Replay with a save and restore after some batches.

    Args:
        arrived: Positions with an ``arrival`` column.
        hold: Reorder hold.
        length: Batch length.
        stops: Indices of the batches after which to stop.
        drain: Whether to drain output at each stop.

    Returns:
        tuple: Events by detector in batch order, every incident
        transition in order, and the final counts.
    """
    stream, parts = MicroBatchStream(INPUTS, hold), []
    for index, (now, part) in enumerate(batches(arrived, length)):
        stream.push(part, now)
        if index in stops:
            if drain:
                parts.append(stream.drain())
            stream = through_files(stream)
    stream.flush(now=arrived["arrival"].max() + length)
    parts.append(stream.drain())
    events = {
        name: pl.concat([part[0][name] for part in parts]).sort(order)
        for name, order in ORDER.items()
    }
    return events, [i for part in parts for i in part[1]], stream.counts


@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    every=st.integers(min_value=20, max_value=6 * 60),
    seed=st.integers(min_value=0, max_value=10_000),
    name=st.sampled_from(["ideal", "terrestrial", "backhaul"]),
    stops=st.sets(st.integers(min_value=0, max_value=200), min_size=1, max_size=4),
    drain=st.booleans(),
)
def test_restore_and_continue_equals_uninterrupted(every, seed, name, stops, drain):
    """Stopping after any batches, saving, restoring and carrying on gives
    the same events and incidents, whether or not output is drained at
    each stop."""
    scene = SCENARIOS[name]
    arrived = arrivals(DATA, scene, seed)
    length = timedelta(minutes=every)
    whole = replay(arrived, INPUTS, scene.hold, length)
    events, incidents, counts = stopped_run(arrived, scene.hold, length, stops, drain)
    for detector, expected in whole.events().items():
        assert_frame_equal(events[detector], expected)
    assert incidents == whole.incidents
    assert counts == whole.counts


def test_relayed_repeat_after_restore_is_still_a_repeat():
    """A moving position heard again after a restart is still classed as a
    relayed repeat, as it would be without the restart."""
    track = frame(Track("relay", -68.0, 42.0).leg(60, 8.0))
    early = pl.col("timestamp") < T0 + timedelta(minutes=30)
    first = track.filter(early)
    # The same message relayed an hour later: same position, course and
    # speed under a later timestamp.
    relayed = track.slice(3, 1).with_columns(timestamp=T0 + timedelta(hours=2))
    second = pl.concat([track.filter(early.not_()), relayed])
    tails = []
    for stop in (False, True):
        stream = MicroBatchStream(INPUTS, timedelta(0))
        stream.push(first, now=T0)
        if stop:
            stream = through_files(stream)
        stream.push(second, now=T0 + timedelta(hours=2))
        tails.append(snapshot(stream)[0]["tail"])
    assert "repeat" in tails[0]["point_status"].to_list()
    assert_frame_equal(tails[1], tails[0])


def test_restore_before_any_position():
    """A stream saved before anything arrived restores and runs."""
    scene = SCENARIOS["ideal"]
    arrived = arrivals(DATA, scene)
    stream = through_files(MicroBatchStream(INPUTS, scene.hold))
    for now, part in batches(arrived, timedelta(hours=1)):
        stream.push(part, now)
    stream.flush(now=arrived["arrival"].max())
    whole = replay(arrived, INPUTS, scene.hold, timedelta(hours=1))
    assert stream.incidents == whole.incidents


def test_drain_empties_the_output():
    """After a drain the stream holds no settled episodes or incidents."""
    stream = replay(
        arrivals(DATA, SCENARIOS["ideal"]), INPUTS, timedelta(0), timedelta(hours=1)
    )
    events, incidents = stream.drain()
    assert sum(frame.height for frame in events.values()) > 0 and incidents
    assert all(frame.is_empty() for frame in stream.events().values())
    assert not stream.incidents
    frames, meta = snapshot(stream)
    assert not [name for name in frames if name.startswith("settled_")]
    assert not meta["incidents"]


def test_other_format_is_refused():
    """A checkpoint written in another layout is refused, not misread."""
    frames, meta = snapshot(MicroBatchStream(INPUTS, timedelta(0)))
    with pytest.raises(ValueError, match="format"):
        restore(frames, {**meta, "format": 0}, INPUTS)
