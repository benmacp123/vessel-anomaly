"""Checkpoints: a stream's whole state as polars frames plus JSON, so a
service can stop between batches and carry on in another process.

A checkpoint is the frames (positions held for reordering, vessel tails,
mooring flags, repeat keys, settled episodes not yet drained) and one
JSON-ready dict for everything else (counts, watermark, segmenter,
moorings, incident states, episode keys already settled, incidents not yet
drained). The caller writes the frames as Parquet and the dict as JSON;
this module does no I/O. Pickle is not used: loading it can run code, and
it ties saved state to the class layout.

The reference layers and thresholds (``Inputs``) are not saved. The caller
passes the same ones on restore, as it did when the stream started.

Restoring and continuing gives the same events and incidents as never
stopping (``tests/test_checkpoint.py``). The last detection pass's
unsettled episodes are not saved: every pass recomputes them before using
them.
"""

# The stream's state is private to it; this module is its serialiser, kept
# apart so the stream module stays about detection.
# pylint: disable=protected-access

from datetime import datetime, timedelta

import polars as pl

from vessels.core.gaps import Moorings
from vessels.core.lifecycle import Incident
from vessels.core.segment import Segmenter
from vessels.stream.micro import SCHEMAS, Inputs, MicroBatchStream

# Raised when the saved layout changes, so an old checkpoint is refused
# rather than misread.
FORMAT = 1
MICROSECOND = timedelta(microseconds=1)


def _time(value: datetime | None) -> str | None:
    """ISO 8601 text for a time, or None.

    Args:
        value: A time, or None.

    Returns:
        str | None: The text, or None.
    """
    return None if value is None else value.isoformat()


def _parse(text: str | None) -> datetime | None:
    """Read a time written by ``_time``.

    Args:
        text: ISO 8601 text, or None.

    Returns:
        datetime | None: The time, or None.
    """
    return None if text is None else datetime.fromisoformat(text)


def snapshot(stream: MicroBatchStream) -> tuple[dict[str, pl.DataFrame], dict]:
    """Capture a stream's state between batches.

    Args:
        stream: The stream; it is not changed.

    Returns:
        tuple: Frames by name (a frame the stream has not started is
        absent) and the JSON-ready rest.
    """
    frames = {
        name: frame
        for name, frame in (
            ("buffer", stream._buffer),
            ("tail", stream._tail),
            ("flags", stream._flags),
        )
        if frame is not None
    }
    # Repeat keys are one string per position ever heard, too many for
    # JSON; sorted so the same state always writes the same file.
    frames["seen"] = pl.DataFrame(
        [(vessel, key) for vessel, keys in stream._seen.items() for key in keys],
        schema={"vessel_id": pl.String, "key": pl.String},
        orient="row",
    ).sort("vessel_id", "key")
    for detector, parts in stream._settled.items():
        if parts:
            frames[f"settled_{detector}"] = pl.concat(parts)
    meta = {
        "format": FORMAT,
        "hold_us": stream.hold // MICROSECOND,
        "counts": dict(stream.counts),
        "latest": _time(stream._latest),
        "released": _time(stream._released),
        "segmenter": stream._segmenter.to_dict(),
        "moorings": stream._moorings.state,
        "states": stream._states,
        # JSON has no tuple keys: each entry is the key's parts and a time.
        "emitted_until": [
            [list(key), _time(end)] for key, end in stream._emitted_until.items()
        ],
        "incidents": [incident.to_dict() for incident in stream.incidents],
    }
    return frames, meta


def restore(
    frames: dict[str, pl.DataFrame], meta: dict, inputs: Inputs
) -> MicroBatchStream:
    """Rebuild a stream from a ``snapshot``.

    Args:
        frames: ``snapshot`` frames, read back from Parquet.
        meta: ``snapshot`` dict, read back from JSON.
        inputs: The reference layers and thresholds the stream ran with.

    Returns:
        MicroBatchStream: A stream that continues as the saved one would.

    Raises:
        ValueError: If the checkpoint was written in another format.
    """
    if meta.get("format") != FORMAT:
        raise ValueError(f"checkpoint format {meta.get('format')}, expected {FORMAT}")
    stream = MicroBatchStream(inputs, meta["hold_us"] * MICROSECOND)
    stream.counts = dict(meta["counts"])
    stream._buffer = frames.get("buffer")
    stream._tail = frames.get("tail")
    stream._flags = frames.get("flags")
    stream._latest = _parse(meta["latest"])
    stream._released = _parse(meta["released"])
    stream._segmenter = Segmenter.from_dict(meta["segmenter"])
    stream._moorings = Moorings(inputs.ports.projection, meta["moorings"])
    stream._states = meta["states"]
    stream._emitted_until = {
        tuple(key): _parse(end) for key, end in meta["emitted_until"]
    }
    seen: dict[str, set[str]] = {}
    for vessel, key in frames["seen"].iter_rows():
        seen.setdefault(vessel, set()).add(key)
    stream._seen = seen
    for detector in SCHEMAS:
        part = frames.get(f"settled_{detector}")
        stream._settled[detector] = [] if part is None else [part]
    stream.incidents = [Incident.from_dict(row) for row in meta["incidents"]]
    return stream
