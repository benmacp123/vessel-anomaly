"""Score the detectors on one block: real alert rates, planted recall,
injection realism and detection delay.

``make evaluate`` runs the tuning block, to develop and check the scoring
(its results are committed as that check). ``make score`` runs the held-out
block once, after the owner has committed ``config/frozen.json`` and its
hash (``make freeze``): it refuses to run if the code's thresholds differ
from the committed ones, or if held-out results already exist. Only counts
and rates are written, never event rows (a hashed ID with exact times and
positions would let a reader rebuild a track). Writes, under
``results/evaluation/``, for ``<block>``:

- ``<block>_alerts.csv``: real-data alert rates with Poisson intervals:
  gaps by variant (rule, without the coverage mask, GFW's 50 nm rule),
  class and distance to the coverage edge; closures per 1,000
  fishing-vessel-hours nearby; loitering (8 hours, and 4 as sensitivity)
  and encounters per 1,000 vessel-days; each detector per day;
- ``<block>_recall.csv``: planted cases hit, by kind and label, and for
  positives by class and reception density, with Wilson intervals;
- ``<block>_scores.csv``: confusion counts and scores on planted cases;
- ``<block>_realism.csv`` and ``<block>_realism_auc.csv``: planted against
  real gaps;
- ``<block>_delay.csv``: hours from when a planted positive could first
  alert to the stream's first incident for it (backhaul arrivals, hourly
  batches);
- ``<block>_run.json``: the configuration hash, commit, sizes and run time.
"""

import argparse
import json
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from vessels.core.closure_fishing import DETECTOR as CLOSURE
from vessels.core.encounters import DETECTOR as ENCOUNTER
from vessels.core.gaps import DETECTOR as GAP
from vessels.core.loitering import DETECTOR as LOITER
from vessels.core.loitering import LoiterRules
from vessels.evaluation import alerts, config, injection, realism
from vessels.evaluation.inputs import load, reception_days
from vessels.evaluation.metrics import confusion, delay_summary, scores
from vessels.sim.arrival import SCENARIOS, arrivals
from vessels.sim.inject import InjectionRules
from vessels.stream.micro import Inputs, replay, run_batch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "evaluation"
KINDS = (GAP, CLOSURE, ENCOUNTER, LOITER)
ARRIVAL = injection.ARRIVAL
EVERY = timedelta(hours=injection.EVERY_HOURS)
ARRIVAL_SEED = injection.ARRIVAL_SEED


def commit() -> str | None:
    """The checked-out commit, if git can say.

    Returns:
        str | None: The commit hash, with ``-dirty`` when files changed.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return None
    return f"{head}-dirty" if dirty else head


def real_alerts(positions: pl.DataFrame, inputs: Inputs) -> tuple[pl.DataFrame, dict]:
    """Alert rates on the block as recorded, with nothing planted.

    Args:
        positions: The block's positions.
        inputs: Detector inputs.

    Returns:
        tuple[pl.DataFrame, dict]: Rate rows, and the batch events by
        detector (gaps are reused by the realism check).
    """
    events = run_batch(positions, inputs)
    days = alerts.vessel_days(positions)
    n_days = positions["timestamp"].dt.date().n_unique()
    sensitivity = LoiterRules(min_hours=config.LOITER_SENSITIVITY_HOURS)
    loiter_4h = run_batch(positions, replace(inputs, loiter_rules=sensitivity))[LOITER]
    gaps = alerts.gap_variants(events[GAP], inputs.shore)["rule"]
    areas = {
        r.area_id: r.geometry
        for r in inputs.closures
        if r.kind == "closure"
        and (
            r.area_id in config.SCORED
            or r.area_id in set(events[CLOSURE]["closure_id"])
        )
    }
    rows = alerts.gap_rates(events[GAP], inputs.grid, inputs.shore, days)
    rows += alerts.closure_rates(
        events[CLOSURE], positions, areas, inputs.ports.projection
    )
    rows += alerts.per_vessel_day_rates(LOITER, "8 h", events[LOITER], days)
    rows += alerts.per_vessel_day_rates(LOITER, "4 h", loiter_4h, days)
    rows += alerts.per_vessel_day_rates(ENCOUNTER, "rule", events[ENCOUNTER], days)
    scored = events[CLOSURE].filter(
        pl.col("closure_id").is_in(list(config.SCORED))
        & (pl.col("reason") == "closure")
    )
    rows += [
        alerts.region_rate(GAP, "rule", gaps, n_days),
        alerts.region_rate(CLOSURE, "scored closures", scored, n_days),
        alerts.region_rate(LOITER, "8 h", events[LOITER], n_days),
        alerts.region_rate(ENCOUNTER, "rule", events[ENCOUNTER], n_days),
    ]
    return pl.DataFrame(rows, schema=alerts.ALERT_SCHEMA), events


def planted_gaps(scenario: injection.Scenario) -> pl.DataFrame:
    """Detected gap rows that overlap a planted positive.

    Args:
        scenario: The gap scenario.

    Returns:
        pl.DataFrame: Gap rows, one per planted positive found.
    """
    positives = scenario.truth.filter(pl.col("label") == "positive").select(
        "vessel_id", t_start="start", t_end="end"
    )
    return (
        scenario.found.join(positives, on="vessel_id")
        .filter(
            (pl.col("start") <= pl.col("t_end")) & (pl.col("end") >= pl.col("t_start"))
        )
        .unique(["vessel_id", "start"], maintain_order=True)
        .drop("t_start", "t_end")
    )


def score_tables(truths: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Recall by stratum, and confusion scores, over every kind.

    Args:
        truths: Matched truth with density bands, by kind.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Recall and scores tables.
    """
    recall, rows = [], []
    for kind, truth in truths.items():
        overall = injection.recall_table(truth, ["label"]).with_columns(
            transceiver=pl.lit("all"), density_band=pl.lit("all")
        )
        positives = truth.filter(pl.col("label") == "positive")
        by_class = injection.recall_table(
            positives, ["label", "transceiver"]
        ).with_columns(density_band=pl.lit("all"))
        by_band = injection.recall_table(
            positives, ["label", "transceiver", "density_band"]
        )
        for table in (overall, by_class, by_band):
            recall.append(
                table.with_columns(kind=pl.lit(kind)).select(
                    "kind",
                    "label",
                    "transceiver",
                    "density_band",
                    "planted",
                    "hit",
                    "share",
                    "low",
                    "high",
                )
            )
        negatives = truth.filter(pl.col("label") != "positive")
        counts = confusion(positives["hit"], negatives["hit"])
        rows.append(
            {"kind": kind, **counts}
            | {k: None if v is None else round(v, 3) for k, v in scores(counts).items()}
        )
    return pl.concat(recall), pl.DataFrame(rows)


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Score one block and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", choices=("tuning", "heldout"), required=True)
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="skip the stream replays (no delay table)",
    )
    args = parser.parse_args()
    block = args.block
    if block == "heldout":
        # The one-shot rule: thresholds must be the committed, frozen ones,
        # and the held-out block is scored once.
        config_hash, frozen = config.check(), True
        if (OUT / "heldout_run.json").exists():
            raise SystemExit(
                "held-out results exist; the held-out block is scored once"
            )
    else:
        config_hash = config.digest(config.frozen())
        try:
            frozen = config.check() == config_hash
        except RuntimeError:
            frozen = False
    timings, clock = {}, time.monotonic()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    positions, inputs = load(block)
    table, events = real_alerts(positions, inputs)
    timings["real"] = round(time.monotonic() - clock, 1)

    tuning_positions, tuning_inputs = (
        (positions, inputs) if block == "tuning" else load("tuning")
    )
    tuning_gaps = (
        events[GAP]
        if block == "tuning"
        else run_batch(tuning_positions, tuning_inputs)[GAP]
    )
    lengths = injection.gap_lengths(tuning_gaps)
    annotated = injection.annotate(positions, inputs)
    reception = reception_days(block)
    rules = InjectionRules()
    truths, delays, scenarios = {}, [], {}
    for kind in KINDS:
        scenario = injection.run_scenario(
            kind, annotated, inputs, reception, lengths, config.SCORED, rules
        )
        scenarios[kind] = scenario
        truths[kind] = injection.with_density(scenario.truth, annotated)
        timings[kind] = round(time.monotonic() - clock, 1)
        if args.no_stream:
            continue
        arrival = SCENARIOS[ARRIVAL]
        stream = replay(
            arrivals(scenario.positions, arrival, ARRIVAL_SEED),
            scenario.inputs,
            arrival.hold,
            EVERY,
        )
        found = injection.delays(
            scenario.truth,
            stream.incidents,
            kind,
            injection.ready_hours(kind, scenario.inputs),
        )
        hit = scenario.truth.filter(pl.col("label") == "positive")["hit"]
        delays.append(
            {"kind": kind, "positives": found.height, "batch_hits": int(hit.sum())}
            | {
                k: v if v is None or isinstance(v, int) else round(v, 2)
                for k, v in delay_summary(found["delay_hours"]).items()
            }
            | {"alerted": found["delay_hours"].is_not_null().sum()}
        )
        timings[f"{kind}_stream"] = round(time.monotonic() - clock, 1)

    recall, score = score_tables(truths)
    grid_density = annotated.select("h3_cell", "transceiver", "density").unique(
        ["h3_cell", "transceiver"]
    )
    low, high = injection.LENGTH_RANGE_HOURS
    real = events[GAP].filter(
        pl.col("covered")
        & pl.col("away_from_port")
        & pl.col("hours").is_between(low, high)
    )
    planted = realism.features(planted_gaps(scenarios[GAP]), grid_density)
    real_features = realism.features(real, grid_density)
    distribution = realism.distributions(planted, real_features)
    auc = pl.DataFrame([realism.discriminator(planted, real_features, rules.seed)])

    OUT.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT / f"{block}_alerts.csv")
    recall.write_csv(OUT / f"{block}_recall.csv")
    score.write_csv(OUT / f"{block}_scores.csv")
    distribution.write_csv(OUT / f"{block}_realism.csv")
    auc.write_csv(OUT / f"{block}_realism_auc.csv")
    if delays:
        pl.DataFrame(delays).write_csv(OUT / f"{block}_delay.csv")
    days = alerts.vessel_days(positions)
    run = {
        "block": block,
        "config_sha256": config_hash,
        "config_frozen": frozen,
        "commit": commit(),
        "started_utc": started,
        "positions": positions.height,
        "vessels": positions["vessel_id"].n_unique(),
        "vessel_days": days,
        "injection": asdict(rules),
        "arrival": None if args.no_stream else ARRIVAL,
        "seconds": timings | {"total": round(time.monotonic() - clock, 1)},
    }
    (OUT / f"{block}_run.json").write_text(
        json.dumps(run, indent=2, default=str) + "\n"
    )
    print(json.dumps(run["seconds"]))


if __name__ == "__main__":
    main()
