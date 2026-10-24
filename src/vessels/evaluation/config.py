"""The frozen configuration: every threshold the held-out block is scored with.

Decided 2026-10-24: the literature-sourced and provisional values are
frozen as they stand, with no search on the tuning block. ``frozen()``
builds the configuration from the code's own defaults, so the committed
``config/frozen.json`` cannot drift from the code unnoticed:
``tests/test_config.py`` fails until the file is rewritten with
``make freeze``, which shows any change in the diff. The configuration also
pins the SHA-256 of every committed input the detectors read (reference
layers, the tuning coverage grid, the extract checksums).

``make freeze`` writes ``config/frozen.sha256`` too. It is committed once,
before the held-out block is read, and ``scripts/score_heldout.py`` refuses
to run if the configuration's hash differs from it.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from vessels.core import cleaning, coverage, distance, segment
from vessels.core.closure_fishing import TOWING_KN
from vessels.core.encounters import EncounterRules
from vessels.core.events import EventRules
from vessels.core.gaps import GapRules
from vessels.core.loitering import LoiterRules
from vessels.evaluation import alerts, injection
from vessels.sim.inject import InjectionRules

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "frozen.json"
RECORDED = ROOT / "config" / "frozen.sha256"
# Committed files the detectors read, relative to the repo root.
INPUTS = (
    "data/reference/us/closures.geojson",
    "data/reference/us/anchorages.csv",
    "data/reference/us/land.geojson",
    "data/extract/us/SHA256SUMS",
    "results/coverage/tuning_coverage.parquet",
)
# Decided 2026-10-24 (PLAN.md "Scored closures"): the five headline
# candidates, kept as headline only with at least MIN_VESSELS vessels and
# MIN_EVENTS events on the tuning block; the rule was fixed before any
# held-out read.
HEADLINE = (
    "wgom_hma",
    "eastern_maine_hma",
    "gsc_hma",
    "stellwagen_dhra",
    "jeffreys_bank_hma",
)
MIN_VESSELS = 3
MIN_EVENTS = 5
# The candidates that passed that rule on the tuning block
# (results/detectors/tuning_closures.csv); Jeffreys Bank moved to report
# only. The held-out block is scored on these.
SCORED = ("wgom_hma", "eastern_maine_hma", "gsc_hma", "stellwagen_dhra")
# Loitering's sensitivity threshold (later GFW work uses 4 hours); the
# headline is LoiterRules' 8 hours.
LOITER_SENSITIVITY_HOURS = 4.0


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes.

    Args:
        path: The file.

    Returns:
        str: Lower-case hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen() -> dict:
    """Build the configuration from the code's current defaults.

    Returns:
        dict: Plain JSON values, grouped by stage.
    """
    return {
        "cleaning": {
            "max_sog": cleaning.MAX_SOG,
            "noise_sog": list(cleaning.NOISE_SOG),
            "noise_tolerance": cleaning.NOISE_TOLERANCE,
        },
        "segmentation": {
            "max_hours": segment.MAX_HOURS,
            "max_knots": segment.MAX_KNOTS,
            "penalty_hours": segment.PENALTY_HOURS,
            "hours_exp": segment.HOURS_EXP,
            "buffer_hours": segment.BUFFER_HOURS,
            "penalty_speed": segment.PENALTY_SPEED,
            "shape_factor": segment.SHAPE_FACTOR,
            "ambiguity_factor": segment.AMBIGUITY_FACTOR,
            "short_seg_threshold": segment.SHORT_SEG_THRESHOLD,
            "transponder_mismatch_weight": segment.TRANSPONDER_MISMATCH_WEIGHT,
            "very_slow": segment.VERY_SLOW,
            "max_open_segments": segment.MAX_OPEN_SEGMENTS,
        },
        "coverage": {"max_long_share": coverage.MAX_LONG_SHARE},
        "anchorages": {
            "radius_km": distance.RADIUS_KM,
            "min_hours": distance.MIN_HOURS,
            "resolution": distance.RESOLUTION,
            "min_vessels": distance.MIN_VESSELS,
            "max_shore_km": distance.MAX_SHORE_KM,
        },
        "closure_fishing": {
            "towing_kn": list(TOWING_KN),
            "events": asdict(EventRules()),
            "headline_candidates": list(HEADLINE),
            "min_vessels": MIN_VESSELS,
            "min_events": MIN_EVENTS,
            "scored": list(SCORED),
        },
        "gaps": asdict(GapRules()),
        "loitering": asdict(LoiterRules())
        | {"sensitivity_hours": LOITER_SENSITIVITY_HOURS},
        "encounters": asdict(EncounterRules()),
        "evaluation": {
            "injection": asdict(InjectionRules()),
            "gap_length_range_hours": list(injection.LENGTH_RANGE_HOURS),
            "density_bands": list(injection.DENSITY_BANDS),
            "match_slack_minutes": injection.SLACK_MINUTES,
            "edge_bands_km": list(alerts.EDGE_BANDS_KM),
            "near_closure_km": alerts.NEAR_KM,
            "max_step_hours": alerts.MAX_STEP_HOURS,
            "gfw_offshore_nm": alerts.GFW_OFFSHORE_NM,
            "gfw_min_hours": alerts.GFW_MIN_HOURS,
            "arrival": injection.ARRIVAL,
            "batch_hours": injection.EVERY_HOURS,
            "arrival_seed": injection.ARRIVAL_SEED,
        },
        "inputs": {name: sha256_file(ROOT / name) for name in INPUTS},
    }


def dumps(config: dict) -> str:
    """Serialise a configuration in one canonical form.

    Args:
        config: ``frozen()`` output.

    Returns:
        str: Indented JSON with sorted keys and a final newline.
    """
    return json.dumps(config, indent=2, sort_keys=True) + "\n"


def digest(config: dict) -> str:
    """SHA-256 of a configuration's canonical form.

    Args:
        config: ``frozen()`` output.

    Returns:
        str: Lower-case hex digest, equal to ``sha256sum config/frozen.json``.
    """
    return hashlib.sha256(dumps(config).encode()).hexdigest()


def write(path: Path = CONFIG, recorded: Path = RECORDED) -> str:
    """Write the configuration and its hash (``make freeze``).

    Args:
        path: Where the configuration goes.
        recorded: Where its hash goes, in ``sha256sum`` format.

    Returns:
        str: The hash.
    """
    config = frozen()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(config))
    value = digest(config)
    recorded.write_text(f"{value}  {path.name}\n")
    return value


def check(path: Path = CONFIG, recorded: Path = RECORDED) -> str:
    """Confirm the code, the committed configuration and its hash agree.

    Args:
        path: The committed configuration.
        recorded: The committed hash.

    Returns:
        str: The hash, for results files to record.

    Raises:
        RuntimeError: If the configuration is missing, differs from the
            code's defaults, or its hash differs from the recorded one.
    """
    if not path.exists() or not recorded.exists():
        raise RuntimeError(f"{path} or {recorded} missing: run make freeze")
    config = frozen()
    if path.read_text() != dumps(config):
        raise RuntimeError(f"code defaults differ from {path}")
    value = digest(config)
    if recorded.read_text().split()[0] != value:
        raise RuntimeError(f"{path} hash differs from {recorded}")
    return value


if __name__ == "__main__":
    print(write())
