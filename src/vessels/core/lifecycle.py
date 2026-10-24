"""Alerts and their lifecycle: one incident per episode, not one per
position.

Adapted from collar-telemetry @ v0.1.0 (``telemetry/core/alerts.py``). The
stream reports **transitions**: an ``Alert`` with status ``open`` when an
episode is first seen (a gap passes six hours, a tow in a closure passes
the persistence rule) and ``resolved`` when it is settled. ``AlertLifecycle``
turns those into **incidents**, which are what an operator is told about.

Changes from collar-telemetry:

- **Keyed by episode, not detector.** A vessel can have several episodes
  of one detector open at once (two closures, two transmitters), so each
  alert names its episode, and the episode's ID is the detector's
  ``event_id``. That ID already depends only on the track and start time,
  so a retried invocation produces the same incident.
- **No linked incident.** Collar's ``potential_death`` (mortality plus a
  silent collar) has no AIS counterpart here.

Kept from collar-telemetry:

- **One incident per episode.** A duplicate open and a resolve with
  nothing open are dropped and counted, so a misbehaving caller shows up.
- **Flaps reopen the same incident.** A resolved episode is remembered for
  its detector's ``merge_within``; opening it again in that time reopens
  the incident (``flaps`` counted). Loitering needs this: an event can be
  withdrawn when its mean distance from shore drops under 20 nm, then
  qualify again as the vessel drifts back out.
- **Late resolves are flagged**: a resolve whose evidence predates the open
  gets resolution ``late`` so evaluation can count them.
- **Two clocks.** ``opened_at`` and ``resolved_at`` are evidence times;
  ``raised_at`` and ``resolve_raised_at`` are when the stream processed the
  transition, passed in as ``now``. Detection delay is measured on the
  processing clock.

State is plain JSON, one dict per vessel.
"""

from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime, timedelta

OPEN = "open"
RESOLVED = "resolved"
LATE = "late"


@dataclass(frozen=True)
class Alert:
    """A transition for one episode: it opened or was settled.

    Attributes:
        vessel_id: Vessel the alert is about.
        detector: Detector name.
        episode: The episode's ID (the detector's ``event_id``).
        time: Evidence time: the position or watermark behind it.
        message: Human-readable summary.
        status: ``OPEN`` or ``RESOLVED``.
        reason: Why a resolve happened (for example ``final`` or
            ``withdrawn``); None for an open.
    """

    vessel_id: str
    detector: str
    episode: str
    time: datetime
    message: str
    status: str = OPEN
    reason: str | None = None


@dataclass(frozen=True)
class Incident:  # pylint: disable=too-many-instance-attributes
    """One episode as told to an operator.

    Attributes:
        incident_id: The episode's ID.
        vessel_id: Vessel the incident is about.
        detector: Detector name.
        status: ``OPEN`` or ``RESOLVED``.
        opened_at: Evidence time of the first open.
        message: The opening alert's message.
        resolved_at: Evidence time of the final resolve, or None.
        resolution: The resolve's reason, or ``LATE``; None while open.
        resolve_message: The resolving alert's message, or None.
        flaps: Times the incident resolved and was reopened.
        raised_at: Processing time of the first open, or None.
        resolve_raised_at: Processing time of the final resolve, or None.
    """

    incident_id: str
    vessel_id: str
    detector: str
    status: str
    opened_at: datetime
    message: str
    resolved_at: datetime | None = None
    resolution: str | None = None
    resolve_message: str | None = None
    flaps: int = 0
    raised_at: datetime | None = None
    resolve_raised_at: datetime | None = None

    def to_dict(self) -> dict:
        """JSON-ready form.

        Returns:
            dict: The incident's fields, with times as ISO 8601 text.
        """
        return {
            k: v.isoformat() if isinstance(v, datetime) else v
            for k, v in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Incident":
        """Rebuild an incident saved with ``to_dict``.

        Args:
            data: ``to_dict`` output; extra keys are ignored.

        Returns:
            Incident: The equivalent incident.
        """
        times = ("opened_at", "resolved_at", "raised_at", "resolve_raised_at")
        fields = {f.name: data[f.name] for f in dataclass_fields(cls)}
        for name in times:
            if fields[name]:
                fields[name] = datetime.fromisoformat(fields[name])
        return cls(**fields)


def default_merge_within() -> dict[str, timedelta]:
    """Provisional merge windows, not chosen on the tuning block.

    Only loitering can be withdrawn and qualify again; two days is a
    provisional guess at how long a drifting vessel may take to drift back
    beyond 20 nm. Other detectors never reopen.

    Returns:
        dict[str, timedelta]: Merge window by detector name.
    """
    return {"loitering": timedelta(days=2)}


@dataclass(frozen=True)
class LifecycleConfig:
    """Lifecycle settings.

    Attributes:
        merge_within: How long after a resolve, by detector name, opening
            the same episode reopens its incident.
    """

    merge_within: dict[str, timedelta] = field(default_factory=default_merge_within)


class AlertLifecycle:
    """Turns transitions into incidents, one vessel's state at a time."""

    def __init__(self, config: LifecycleConfig | None = None):
        """Create a lifecycle.

        Args:
            config: Settings; the provisional defaults if omitted.
        """
        self.config = config or LifecycleConfig()

    def apply(
        self, state: dict, alerts: list[Alert], now: datetime | None = None
    ) -> list[Incident]:
        """Apply transitions, in order, to one vessel's incidents.

        Args:
            state: The vessel's lifecycle state, updated in place; starts as
                an empty dict.
            alerts: Transitions for that vessel, in the order raised.
            now: Processing time recorded on each transition, or None.

        Returns:
            list[Incident]: Incidents opened, reopened or resolved, in order.
        """
        if "open" not in state:
            state.update(
                open={}, recent={}, dropped={"duplicate_open": 0, "orphan_resolve": 0}
            )
        events = []
        for alert in alerts:
            if alert.status == OPEN:
                events += self._open(state, alert, now)
            else:
                events += self._resolve(state, alert, now)
        return events

    def _open(self, state: dict, alert: Alert, now: datetime | None) -> list[Incident]:
        """Open an incident, reopen a recent one, or drop a duplicate.

        Args:
            state: The vessel's lifecycle state, updated in place.
            alert: An open transition.
            now: Processing time, or None.

        Returns:
            list[Incident]: The opened or reopened incident, or nothing.
        """
        if alert.episode in state["open"]:
            state["dropped"]["duplicate_open"] += 1
            return []
        recent = state["recent"].pop(alert.episode, None)
        if recent is not None:
            since = alert.time - datetime.fromisoformat(recent["resolved_at"])
            if since <= self.config.merge_within.get(alert.detector, timedelta(0)):
                incident = Incident.from_dict(
                    {
                        **recent,
                        "status": OPEN,
                        "resolved_at": None,
                        "resolution": None,
                        "resolve_message": None,
                        "resolve_raised_at": None,
                        "flaps": recent["flaps"] + 1,
                    }
                )
                state["open"][alert.episode] = incident.to_dict()
                return [incident]
        incident = Incident(
            incident_id=alert.episode,
            vessel_id=alert.vessel_id,
            detector=alert.detector,
            status=OPEN,
            opened_at=alert.time,
            message=alert.message,
            raised_at=now,
        )
        state["open"][alert.episode] = incident.to_dict()
        return [incident]

    def _resolve(
        self, state: dict, alert: Alert, now: datetime | None
    ) -> list[Incident]:
        """Resolve an open incident, remembering it if it may reopen.

        Args:
            state: The vessel's lifecycle state, updated in place.
            alert: A resolved transition.
            now: Processing time, or None.

        Returns:
            list[Incident]: The resolved incident, or nothing if none was
            open.
        """
        record = state["open"].pop(alert.episode, None)
        if record is None:
            state["dropped"]["orphan_resolve"] += 1
            return []
        late = alert.time < datetime.fromisoformat(record["opened_at"])
        incident = Incident.from_dict(
            {
                **record,
                "status": RESOLVED,
                "resolved_at": alert.time.isoformat(),
                "resolution": LATE if late else alert.reason,
                "resolve_message": alert.message,
                "resolve_raised_at": now.isoformat() if now else None,
            }
        )
        if self.config.merge_within.get(alert.detector, timedelta(0)) > timedelta(0):
            state["recent"][alert.episode] = incident.to_dict()
        return [incident]


def open_incidents(state: dict) -> list[Incident]:
    """Incidents still open for a vessel.

    Args:
        state: The vessel's lifecycle state.

    Returns:
        list[Incident]: Open incidents, ordered by ID.
    """
    return [
        Incident.from_dict(state["open"][key]) for key in sorted(state.get("open", {}))
    ]
