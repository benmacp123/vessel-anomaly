"""Tests for the service: the processor over a spool gives the batch output,
and a crash at any step of a tick, followed by a restart, loses and
duplicates nothing."""

import json
import os
from datetime import timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cloud.service import Processor
from cloud.sink import FileSink, feature
from cloud.spool import DONE, END, SpoolSource, publish
from cloud.store import DirectoryStore, Saved, StoreLocked
from golden import inputs_for, positions
from vessels.sim.arrival import SCENARIOS, arrivals
from vessels.stream.micro import ENCOUNTER, GAP, batches, run_batch

DATA = positions("closure_tow", "loiter_drift", "dark_gap", "mooring", "meeting")
INPUTS = inputs_for(DATA)
BATCH = run_batch(DATA, INPUTS)
SCENE = SCENARIOS["backhaul"]
ARRIVED = arrivals(DATA, SCENE, 3)
EVERY = timedelta(hours=1)
# Files moved into the spool before each tick.
STEP = 7


class Crash(Exception):
    """A simulated crash."""


def processor(root):
    """A processor over a spool, store and sink under one directory.

    Args:
        root: The directory.

    Returns:
        Processor: The processor.
    """
    return Processor(
        SpoolSource(root / "spool"),
        DirectoryStore(root / "state", compact_after=3),
        FileSink(root / "alerts"),
        INPUTS,
        SCENE.hold,
    )


def run(root, crash=None, regroup=False):
    """Publish into a staging directory, feed the spool ``STEP`` files at a
    time with a tick after each, and restart after a crash.

    Args:
        root: The working directory.
        crash: None, or ``(tick, part, method, after)``: at that tick,
            make that part's method raise, before or after it does its
            work. The crashed processor is closed (as a dying process
            drops its lock) and a new one started.
        regroup: After a crash, wait for the next files before ticking
            again, so the redone tick reads more than the crashed one;
            otherwise tick again at once over the same files.

    Returns:
        tuple: The final processor, and the sink's events and incidents up
        to the committed tick.
    """
    staging = root / "staging"
    publish(ARRIVED, staging, EVERY)
    files = sorted(staging.glob("*.parquet")) + [staging / END]
    spool = root / "spool"
    proc = processor(root)
    for index in range(0, len(files), STEP):
        for path in files[index : index + STEP]:
            os.replace(path, spool / path.name)
        if crash and crash[0] == proc.tick_number + 1:
            _, part, method, after = crash
            original = getattr(getattr(proc, part), method)

            def boom(*args, original=original, after=after):
                """Raise before or after the real call."""
                if after:
                    original(*args)
                raise Crash

            setattr(getattr(proc, part), method, boom)
            with pytest.raises(Crash):
                proc.tick()
            proc.close()
            proc = processor(root)
            crash = None
            # Nothing a reader of the parts could see may be redone.
            parts = (root / "alerts" / "parts").iterdir()
            assert all(int(p.name[:8]) <= proc.tick_number for p in parts)
            if regroup:
                continue
        proc.tick()
    assert proc.finished
    return proc, *proc.sink.read(proc.tick_number)


def event_ids(root):
    """Event IDs in every alert part, repeats included.

    Args:
        root: The working directory.

    Returns:
        list[str]: The IDs.
    """
    parts = sorted((root / "alerts" / "parts").glob("*.parquet"))
    return [i for p in parts for i in pl.read_parquet(p)["event_id"]]


@pytest.fixture(scope="module", name="whole")
def fixture_whole(tmp_path_factory):
    """An uninterrupted run."""
    root = tmp_path_factory.mktemp("whole")
    return root, *run(root)


def test_service_equals_batch(whole):
    """The service's committed alerts equal the batch detectors' output, and
    every spool file ends up read."""
    root, proc, events, _ = whole
    for name, expected in BATCH.items():
        assert_frame_equal(events[name], expected, rel_tol=1e-12, abs_tol=1e-12)
    assert proc.stream.counts["late"] == 0
    assert not list((root / "spool").glob("*.parquet"))
    windows = sum(1 for _ in batches(ARRIVED, EVERY))
    assert len(list((root / "spool" / DONE).glob("*.parquet"))) == windows


CRASHES = [
    ("sink", "write", True),
    ("store", "_commit", False),
    ("store", "_commit", True),
    ("sink", "publish", True),
    ("source", "mark", False),
]


@pytest.mark.parametrize("part,method,after", CRASHES)
def test_crash_and_restart_loses_and_duplicates_nothing(
    tmp_path, whole, part, method, after
):
    """A crash at any step of a tick, then a restart, gives the same events,
    incidents and counts as never stopping, with each event written once."""
    _, uninterrupted, events, incidents = whole
    proc, got, got_incidents = run(tmp_path, (4, part, method, after))
    for name, expected in events.items():
        assert_frame_equal(got[name], expected)
    assert got_incidents == incidents
    assert proc.stream.counts == uninterrupted.stream.counts
    ids = event_ids(tmp_path)
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("part,method,after", CRASHES)
def test_restart_reading_more_files_than_the_crashed_tick(
    tmp_path, whole, part, method, after
):
    """A redone tick that reads more files than the crashed attempt still
    gives the batch output once, and the same counts."""
    uninterrupted = whole[1]
    proc, got, _ = run(tmp_path, (4, part, method, after), regroup=True)
    for name, expected in BATCH.items():
        assert_frame_equal(got[name], expected, rel_tol=1e-12, abs_tol=1e-12)
    assert proc.stream.counts == uninterrupted.stream.counts
    ids = event_ids(tmp_path)
    assert len(ids) == len(set(ids))


def test_backlog_is_read_in_bounded_ticks(tmp_path):
    """With everything published before the processor starts, a cap on files
    per tick spreads the backlog over many ticks and still gives the batch
    output."""
    publish(ARRIVED, tmp_path / "spool", EVERY)
    proc = Processor(
        SpoolSource(tmp_path / "spool"),
        DirectoryStore(tmp_path / "state"),
        FileSink(tmp_path / "alerts"),
        INPUTS,
        SCENE.hold,
        max_files=25,
    )
    windows = sum(1 for _ in batches(ARRIVED, EVERY))
    assert proc.run(sleep=pytest.fail) == -(-windows // 25)
    for name, expected in proc.sink.read(proc.tick_number)[0].items():
        assert_frame_equal(expected, BATCH[name], rel_tol=1e-12, abs_tol=1e-12)
    proc.close()


def test_restart_after_the_end_does_nothing(whole):
    """A processor restarted on a finished run writes nothing more."""
    root, proc, events, _ = whole
    proc.close()
    again = processor(root)
    assert again.finished and not again.tick()
    for name, frame in again.sink.read(again.tick_number)[0].items():
        assert_frame_equal(frame, events[name])
    again.close()


def test_geojson_holds_every_event_rounded(whole):
    """The map file has one feature per committed event, coordinates to two
    decimal places."""
    root, _, events, _ = whole
    collection = json.loads((root / "alerts" / "alerts.geojson").read_text())
    assert len(collection["features"]) == sum(f.height for f in events.values())
    for item in collection["features"]:
        flat = item["geometry"]["coordinates"]
        flat = flat if isinstance(flat[0], list) else [flat]
        assert all(round(v, 2) == v for point in flat for v in point)


def test_features_draw_gaps_as_jumps_and_encounters_at_the_mean():
    """A gap is a line from where the vessel went dark to where it came back;
    an encounter is a point at the pair's mean position."""
    gap = BATCH[GAP].row(0, named=True)
    line = feature(GAP, gap)["geometry"]
    assert line["type"] == "LineString"
    assert line["coordinates"][1] == [
        round(gap["end_lon"], 2),
        round(gap["end_lat"], 2),
    ]
    meeting = BATCH[ENCOUNTER].row(0, named=True)
    point = feature(ENCOUNTER, meeting)["geometry"]
    assert point["coordinates"] == [
        round(meeting["mean_lon"], 2),
        round(meeting["mean_lat"], 2),
    ]


def keys(*pairs):
    """A repeat-key frame.

    Args:
        *pairs: ``(vessel_id, key)`` pairs.

    Returns:
        pl.DataFrame: The keys.
    """
    return pl.DataFrame(
        list(pairs), schema={"vessel_id": pl.String, "key": pl.String}, orient="row"
    )


def saved(tick, seen):
    """A tick holding only repeat keys.

    Args:
        tick: Tick number.
        seen: Repeat keys.

    Returns:
        Saved: The tick.
    """
    return Saved(tick, {"seen": seen}, {"tick": tick}, [f"{tick}.parquet"], None, False)


def test_repeat_keys_are_written_once_then_compacted(tmp_path):
    """Each tick writes only the keys it added; after the part limit they
    are merged into one file; loading gives every key."""
    store = DirectoryStore(tmp_path, compact_after=3)
    grown = []
    for tick in range(1, 6):
        grown.append(("a", f"k{tick}"))
        store.save(saved(tick, keys(*grown)))
        parts = sorted((tmp_path / "seen").glob("*.parquet"))
        if tick <= 3:
            assert len(parts) == tick
            assert pl.read_parquet(parts[-1]).height == 1
    # Tick 4 found three parts and merged; tick 5 appended one.
    assert [pl.read_parquet(p).height for p in parts] == [4, 1]
    loaded = DirectoryStore(tmp_path).load()
    assert loaded.tick == 5 and loaded.meta == {"tick": 5}
    assert_frame_equal(loaded.frames["seen"], keys(*grown))


def test_lost_repeat_keys_force_a_rewrite(tmp_path):
    """If keys ever went missing, the store rewrites them rather than
    keeping old parts that would bring them back."""
    store = DirectoryStore(tmp_path)
    store.save(saved(1, keys(("a", "k1"), ("a", "k2"))))
    store.save(saved(2, keys(("a", "k2"))))
    assert_frame_equal(
        DirectoryStore(tmp_path).load().frames["seen"], keys(("a", "k2"))
    )


def test_uncommitted_tick_is_ignored_and_tidied(tmp_path):
    """A tick written but not committed is not loaded, and its files are
    removed."""
    store = DirectoryStore(tmp_path)
    store.save(saved(1, keys(("a", "k1"))))

    def boom(tick):
        """Crash before the pointer moves."""
        raise Crash(tick)

    store._commit = boom  # pylint: disable=protected-access
    with pytest.raises(Crash):
        store.save(saved(2, keys(("a", "k1"), ("a", "k2"))))
    loaded = DirectoryStore(tmp_path).load()
    assert loaded.tick == 1
    assert [p.name for p in (tmp_path / "ticks").iterdir()] == ["00000001"]
    assert [p.name for p in (tmp_path / "seen").iterdir()] == ["00000001.parquet"]


def test_ticks_must_follow_the_committed_one(tmp_path):
    """Saving out of order is refused."""
    with pytest.raises(ValueError, match="tick 2 after 0"):
        DirectoryStore(tmp_path).save(saved(2, keys()))


def test_second_processor_is_refused(tmp_path):
    """Only one process may hold a store; it is free again once released."""
    first = DirectoryStore(tmp_path)
    first.lock()
    with pytest.raises(StoreLocked):
        DirectoryStore(tmp_path).lock()
    first.unlock()
    second = DirectoryStore(tmp_path)
    second.lock()
    second.unlock()


def test_publisher_paces_by_arrival_time_and_ends(tmp_path):
    """Files appear in arrival order at the chosen speed, then the end
    marker."""
    waits, now = [], [0.0]

    def sleep(seconds):
        """Record the wait and move the fake clock on."""
        waits.append(seconds)
        now[0] += seconds

    written = publish(
        ARRIVED,
        tmp_path,
        timedelta(hours=2),
        speed=3600.0,
        sleep=sleep,
        clock=lambda: now[0],
    )
    source = SpoolSource(tmp_path)
    names = source.pending()
    assert len(names) == written and source.ended()
    # Two hours of arrival time take two seconds at 3,600 times; windows
    # with no arrivals are skipped, so a wait can span several.
    ends = [end for end, _ in batches(ARRIVED, timedelta(hours=2))]
    assert sum(waits) == pytest.approx((ends[-1] - ends[0]).total_seconds() / 3600)
    assert all(w / 2.0 == pytest.approx(round(w / 2.0)) for w in waits)
    frame = source.read(names)
    assert frame.height == ARRIVED.height
    assert frame["arrival"].is_sorted()


def test_marking_twice_is_harmless(tmp_path):
    """Marking a file already moved does nothing, so a restart can repeat
    it."""
    publish(ARRIVED.head(10), tmp_path, EVERY)
    source = SpoolSource(tmp_path)
    names = source.pending()
    source.mark(names)
    source.mark(names)
    assert not source.pending()
    assert sorted(p.name for p in (tmp_path / DONE).iterdir()) == names


def test_half_written_files_are_not_read(tmp_path):
    """A file still being written is not listed."""
    source = SpoolSource(tmp_path)
    (tmp_path / "20240506T000000Z.parquet.partial").write_bytes(b"")
    assert not source.pending()
