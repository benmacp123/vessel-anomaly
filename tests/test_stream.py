"""Tests for the micro-batch stream: whatever the batch boundaries and
arrival order, settled episodes equal the batch detectors' output, and
incidents open while episodes are still under way."""

from datetime import timedelta

import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from golden import inputs_for, positions
from synthetic import T0, Track, frame
from vessels.sim.arrival import SCENARIOS, ArrivalScenario, arrivals
from vessels.stream.micro import MicroBatchStream, replay, run_batch

DATA = positions("closure_tow", "loiter_drift", "dark_gap", "mooring")
INPUTS = inputs_for(DATA)
BATCH = run_batch(DATA, INPUTS)


def assert_same(stream_events: dict, batch_events: dict) -> None:
    """Every detector's settled episodes equal the batch output.

    Args:
        stream_events: ``MicroBatchStream.events`` output.
        batch_events: ``run_batch`` output.
    """
    for name, expected in batch_events.items():
        # Non-float columns are compared exactly; floats may differ in the
        # last bits when one expression runs over frames of other sizes.
        assert_frame_equal(stream_events[name], expected, rel_tol=1e-12, abs_tol=1e-12)


def test_scenario_exercises_every_detector():
    """The scenario holds at least one episode of each detector, an
    unexplained gap and an explained one."""
    assert BATCH["closure_fishing"].height >= 1
    assert BATCH["loitering"].height >= 1
    gaps = BATCH["gap"]
    assert gaps["unexplained"].any() and not gaps["unexplained"].all()


def test_one_batch_equals_batch():
    """Everything in one batch gives the batch output."""
    stream = replay(
        arrivals(DATA, SCENARIOS["ideal"]), INPUTS, timedelta(0), timedelta(days=9)
    )
    assert_same(stream.events(), BATCH)


@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    every=st.integers(min_value=20, max_value=12 * 60),
    seed=st.integers(min_value=0, max_value=10_000),
    name=st.sampled_from(["ideal", "terrestrial", "backhaul"]),
)
def test_micro_batches_equal_batch(every, seed, name):
    """Any batch length, with out-of-order arrivals and duplicates inside
    the hold, gives the batch output."""
    scene = SCENARIOS[name]
    arrived = arrivals(DATA, scene, seed)
    stream = replay(arrived, INPUTS, scene.hold, timedelta(minutes=every))
    assert stream.counts["late"] == 0
    assert_same(stream.events(), BATCH)


def test_short_hold_drops_late_positions():
    """A hold shorter than the delays drops late positions and counts them."""
    scene = SCENARIOS["short_hold"]
    stream = replay(arrivals(DATA, scene, 1), INPUTS, scene.hold, timedelta(minutes=10))
    assert stream.counts["late"] > 0


def test_dark_vessel_alerted_while_dark():
    """A gap's incident is raised before the vessel reappears."""
    stream = replay(
        arrivals(DATA, SCENARIOS["ideal"]), INPUTS, timedelta(0), timedelta(minutes=30)
    )
    gap = BATCH["gap"].filter(pl.col("hours") > 20).row(0, named=True)
    opened = [i for i in stream.incidents if i.incident_id == gap["event_id"]]
    assert opened[0].raised_at < gap["end"]
    assert opened[-1].resolution in ("unexplained", "explained")


def test_every_settled_episode_has_one_incident():
    """Each settled episode was opened once and resolved once."""
    stream = replay(
        arrivals(DATA, SCENARIOS["backhaul"], 3),
        INPUTS,
        timedelta(minutes=30),
        timedelta(hours=1),
    )
    settled = {e for frame_ in stream.events().values() for e in frame_["event_id"]}
    resolved = [i.incident_id for i in stream.incidents if i.status == "resolved"]
    assert settled <= set(resolved)
    for state in stream._states.values():  # pylint: disable=protected-access
        assert state["dropped"] == {"duplicate_open": 0, "orphan_resolve": 0}


def test_hold_must_cover_delays():
    """The scenarios that promise no late positions have a hold over their
    delay."""
    for scene in SCENARIOS.values():
        if scene.name != "short_hold":
            assert scene.hold >= scene.max_delay
    assert ArrivalScenario("x", timedelta(0)).duplicate_share == 0


def test_loitering_edges_equal_batch():
    """Quarter-hour batches settle neither edge case early or cut short."""
    data = positions("loiter_edges")
    inputs = inputs_for(data)
    expected = run_batch(data, inputs)
    assert expected["loitering"].height == 2
    stream = replay(
        arrivals(data, SCENARIOS["ideal"]), inputs, timedelta(0), timedelta(minutes=15)
    )
    assert_same(stream.events(), expected)


def test_position_at_the_watermark_is_late():
    """A position stamped exactly at a watermark already passed is late, so
    a duplicate cannot slip in after its twin."""
    data = frame(Track("v", -68.0, 42.0).leg(10, 8.0))
    stream = MicroBatchStream(INPUTS, timedelta(0))
    stream.push(data, now=T0)
    stream.push(data.tail(1), now=T0)
    assert stream.counts["late"] == 1
