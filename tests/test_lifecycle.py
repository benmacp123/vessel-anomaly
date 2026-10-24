"""Tests for the alert lifecycle: one incident per episode, flaps, drops,
late resolves and saved state. Adapted from collar-telemetry @ v0.1.0."""

import json
from datetime import timedelta

from synthetic import T0
from vessels.core.lifecycle import (
    LATE,
    OPEN,
    RESOLVED,
    Alert,
    AlertLifecycle,
    Incident,
    open_incidents,
)


def alert(minutes, status=OPEN, episode="v:loitering:1", detector="loitering"):
    """A transition at a time after T0.

    Args:
        minutes: Evidence time, minutes after T0.
        status: ``OPEN`` or ``RESOLVED``.
        episode: Episode ID.
        detector: Detector name.

    Returns:
        Alert: The transition.
    """
    reason = None if status == OPEN else "final"
    return Alert(
        "v", detector, episode, T0 + timedelta(minutes=minutes), "m", status, reason
    )


def test_open_then_resolve():
    """An open and a resolve give one incident, opened then resolved."""
    state = {}
    life = AlertLifecycle()
    opened = life.apply(state, [alert(0)], now=T0 + timedelta(minutes=5))
    resolved = life.apply(state, [alert(60, RESOLVED)], now=T0 + timedelta(hours=2))
    assert [i.status for i in opened + resolved] == [OPEN, RESOLVED]
    assert resolved[0].incident_id == "v:loitering:1"
    assert resolved[0].raised_at == T0 + timedelta(minutes=5)
    assert open_incidents(state) == []


def test_duplicates_and_orphans_are_dropped():
    """A second open and a resolve with nothing open are counted, not
    raised."""
    state = {}
    life = AlertLifecycle()
    life.apply(state, [alert(0), alert(10)])
    life.apply(state, [alert(20, RESOLVED), alert(30, RESOLVED)])
    assert state["dropped"] == {"duplicate_open": 1, "orphan_resolve": 1}


def test_two_episodes_open_together():
    """Two episodes of one detector are separate incidents."""
    state = {}
    found = AlertLifecycle().apply(state, [alert(0), alert(5, episode="v:loitering:2")])
    assert len(found) == 2
    assert len(open_incidents(state)) == 2


def test_reopen_within_window_flaps():
    """A loitering episode opened again within two days of its resolve
    reopens the same incident."""
    state = {}
    life = AlertLifecycle()
    life.apply(state, [alert(0), alert(60, RESOLVED)])
    again = life.apply(state, [alert(120)])
    assert again[0].flaps == 1 and again[0].opened_at == T0


def test_no_reopen_after_window_or_for_gaps():
    """Outside the window, or for a detector without one, an open starts a
    new incident."""
    state = {}
    life = AlertLifecycle()
    life.apply(state, [alert(0), alert(60, RESOLVED)])
    late = life.apply(state, [alert(60 + 3 * 24 * 60)])
    assert late[0].flaps == 0 and late[0].opened_at > T0
    gap = dict(episode="v:gap:1", detector="gap")
    state = {}
    life.apply(state, [alert(0, **gap), alert(60, RESOLVED, **gap)])
    assert life.apply(state, [alert(61, **gap)])[0].flaps == 0


def test_late_resolve_flagged():
    """A resolve whose evidence predates the open is marked late."""
    state = {}
    found = AlertLifecycle().apply(state, [alert(60), alert(0, RESOLVED)])
    assert found[-1].resolution == LATE


def test_state_and_incidents_are_json():
    """State and incidents survive a JSON round trip."""
    state = {}
    found = AlertLifecycle().apply(state, [alert(0)], now=T0)
    state = json.loads(json.dumps(state))
    assert open_incidents(state) == found
    assert Incident.from_dict(json.loads(json.dumps(found[0].to_dict()))) == found[0]
