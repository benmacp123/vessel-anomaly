"""First spike: match Norwegian VMS call signs to AIS MMSIs.

Local only: it reads the git-ignored raw files and writes its working
data to ``data/raw/spike/``. The pass and fail criteria are fixed in the
project plan before running. Steps are subcommands, run in order:

    PYTHONPATH=src python scripts/spike_identity.py vms
    PYTHONPATH=src python scripts/spike_identity.py match
    PYTHONPATH=src python scripts/spike_identity.py assign
    PYTHONPATH=src python scripts/spike_identity.py control
    PYTHONPATH=src python scripts/spike_identity.py types

``vms`` cuts the spike week out of the 2024 VMS archive, keeps fixes
inside the five HAIS order boxes plus a buffer, and drops repeated
reports of the same vessel in the same minute. ``match`` finds, for each
VMS fix, every MMSI with an AIS fix within 500 m and 2 minutes, and
counts matched fixes per (call sign, MMSI). ``assign`` applies the
plan's assignment rules and prints the match rate. ``control`` repeats
both with VMS times shifted by 3 h, to measure chance matches.
``types`` checks the assigned MMSIs against the fishing-only HAIS
delivery.
"""

import argparse
import math
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from vessels.io.fiskeridir import read_vms_zip

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = RAW / "spike"

# The HAIS order (2026-10-23): one week, five boxes of
# (lon_min, lon_max, lat_min, lat_max), each a closure plus about 25 km.
WEEK = [date(2024, 5, 27) + timedelta(days=i) for i in range(7)]
BOXES = {
    "rost": (8.2, 10.3, 67.05, 67.85),
    "traena": (10.3, 11.85, 66.65, 67.25),
    "sula": (7.2, 8.75, 63.65, 64.55),
    "iverryggen": (8.6, 9.95, 64.7, 65.4),
    "more_west": (2.9, 4.2, 62.25, 62.9),
}
# Wide enough that a VMS fix just outside a box still reaches AIS fixes
# just inside it (10 min at 12 kn is about 4 km); HAIS holds nothing
# beyond the boxes, so a wider margin adds only unmatched fixes.
BUFFER_KM = 10.0
KM_PER_DEGREE_LAT = 111.32
EARTH_RADIUS_M = 6_371_008.8
# Match thresholds fixed in the plan before the spike ran.
RADIUS_M = 500.0
WINDOW = timedelta(minutes=2)
MIN_FIXES = 5
MIN_SHARE = 0.8
LEAD_RATIO = 2.0
# The false-match control: far enough that a moving vessel is kilometres
# from its old position, so matches should be strangers or idle vessels.
SHIFT = timedelta(hours=3)


def box_label(boxes: dict, buffer_km: float) -> pl.Expr:
    """Name the first box (with buffer) each fix falls in.

    The buffer in longitude uses the box's poleward edge, where degrees
    are shortest, so it is at least ``buffer_km`` everywhere in the box.
    Røst and Træna overlap once buffered; a fix there takes the box listed
    first.

    Args:
        boxes: Name to (lon_min, lon_max, lat_min, lat_max).
        buffer_km: Margin added on every side.

    Returns:
        pl.Expr: A string column ``box``, null outside every box.
    """
    label = pl.lit(None, dtype=pl.String)
    for name, (lon_min, lon_max, lat_min, lat_max) in reversed(boxes.items()):
        dlat = buffer_km / KM_PER_DEGREE_LAT
        dlon = buffer_km / (KM_PER_DEGREE_LAT * math.cos(math.radians(lat_max)))
        inside = pl.col("lon").is_between(lon_min - dlon, lon_max + dlon) & pl.col(
            "lat"
        ).is_between(lat_min - dlat, lat_max + dlat)
        label = pl.when(inside).then(pl.lit(name)).otherwise(label)
    return label.alias("box")


def cut_vms(vms: pl.DataFrame) -> pl.DataFrame:
    """Keep in-box fixes, one per vessel per minute.

    About 0.5% of the week's rows repeat a call sign and minute, either
    exactly or a few metres apart; the lowest message ID is kept so the
    choice is reproducible.

    Args:
        vms: Rows from ``read_vms_zip``.

    Returns:
        pl.DataFrame: The input columns plus ``box`` (buffered) and
        ``in_box`` (inside a box with no buffer, where HAIS has data),
        sorted by call sign and time.
    """
    return (
        vms.with_columns(
            box_label(BOXES, BUFFER_KM),
            in_box=box_label(BOXES, 0.0).is_not_null(),
        )
        .filter(pl.col("box").is_not_null())
        .sort("message_id")
        .unique(["call_sign", "timestamp"], keep="first", maintain_order=True)
        .sort("call_sign", "timestamp")
    )


def step_vms():
    """Write the spike week's in-box VMS fixes and print their counts."""
    vms = read_vms_zip(RAW / "fiskeridir" / "2024-VMS.csv.zip", WEEK)
    cut = cut_vms(vms)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "vms_week.parquet"
    cut.write_parquet(path, compression="zstd")
    per_call = cut.group_by("call_sign").len()
    print(f"VMS week: {vms.height:,} rows, {vms['call_sign'].n_unique()} call signs")
    print(
        f"In boxes: {cut.height:,} fixes, {per_call.height} call signs, "
        f"{(per_call['len'] >= 5).sum()} with at least 5 fixes"
    )
    print(cut.group_by("box").agg(pl.len(), pl.col("call_sign").n_unique()).sort("box"))
    print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


def load_ais(paths: str) -> pl.DataFrame:
    """Read HAIS position reports.

    HAIS Parquet carries a GEOMETRY column polars cannot read, so DuckDB
    selects the plain columns. Its timestamps are naive UTC.

    Args:
        paths: A glob over the daily HAIS files.

    Returns:
        pl.DataFrame: ``mmsi``, ``timestamp`` (UTC), ``lon``, ``lat``,
        ``sog_kn`` and ``data_source``, sorted by time.
    """
    sql = f"""
        select mmsi, date_time_utc as timestamp, longitude as lon,
               latitude as lat, speed_over_ground::double as sog_kn,
               data_source
        from read_parquet('{paths}')
        order by timestamp
    """
    return (
        # DuckDB 2.0's .pl() needs pyarrow; polars reads the relation
        # through the Arrow stream interface without it.
        pl.DataFrame(duckdb.sql(sql)).with_columns(
            pl.col("mmsi").cast(pl.Int64),
            pl.col("timestamp").dt.replace_time_zone("UTC"),
        )
    )


def unit_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Positions on a sphere of Earth's mean radius, in metres.

    Straight-line (chord) distance between these points differs from
    great-circle distance by under a millimetre at 500 m, and needs no
    choice of map projection across boxes 5 degrees of latitude apart.

    Args:
        lon: Longitudes in degrees.
        lat: Latitudes in degrees.

    Returns:
        np.ndarray: An (n, 3) array of x, y, z.
    """
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    return EARTH_RADIUS_M * np.column_stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)]
    )


def match_fixes(
    vms: pl.DataFrame,
    ais: pl.DataFrame,
    radius_m: float = RADIUS_M,
    window: timedelta = WINDOW,
    shift: timedelta = timedelta(0),
) -> pl.DataFrame:
    """Find, for each VMS fix, the MMSIs with an AIS fix close in space and time.

    AIS is sorted by time, so each VMS fix looks only at the AIS fixes in
    its time window and measures all of their distances directly.

    Args:
        vms: Fixes from ``cut_vms``.
        ais: Reports from ``load_ais``.
        radius_m: Largest distance that counts as a match.
        window: Largest time difference either side.
        shift: Added to every VMS time first; the false-match control
            uses +3 h.

    Returns:
        pl.DataFrame: One row per (VMS fix, MMSI) match: ``message_id``,
        ``call_sign``, ``in_box``, ``mmsi``, the closest distance
        ``distance_m`` and the smallest time difference ``dt_s`` among
        that MMSI's matching AIS fixes.
    """
    ais_t = ais["timestamp"].dt.epoch("s").to_numpy()
    ais_xyz = unit_xyz(ais["lon"].to_numpy(), ais["lat"].to_numpy())
    ais_mmsi = ais["mmsi"].to_numpy()
    vms_t = vms["timestamp"].dt.epoch("s").to_numpy() + int(shift.total_seconds())
    vms_xyz = unit_xyz(vms["lon"].to_numpy(), vms["lat"].to_numpy())
    span = int(window.total_seconds())
    starts = np.searchsorted(ais_t, vms_t - span, side="left")
    ends = np.searchsorted(ais_t, vms_t + span, side="right")
    rows = {"index": [], "mmsi": [], "distance_m": [], "dt_s": []}
    for i, (lo, hi) in enumerate(zip(starts, ends)):
        distance = np.linalg.norm(ais_xyz[lo:hi] - vms_xyz[i], axis=1)
        near = distance <= radius_m
        if not near.any():
            continue
        hits = (
            pl.DataFrame(
                {
                    "mmsi": ais_mmsi[lo:hi][near],
                    "distance_m": distance[near],
                    "dt_s": np.abs(ais_t[lo:hi][near] - vms_t[i]),
                }
            )
            .group_by("mmsi")
            .agg(pl.col("distance_m").min(), pl.col("dt_s").min())
        )
        rows["index"].extend([i] * hits.height)
        for column in ("mmsi", "distance_m", "dt_s"):
            rows[column].extend(hits[column].to_list())
    matches = pl.DataFrame(
        rows,
        schema={
            "index": pl.UInt32,
            "mmsi": pl.Int64,
            "distance_m": pl.Float64,
            "dt_s": pl.Int64,
        },
    )
    fixes = vms.select("message_id", "call_sign", "in_box").with_row_index("index")
    return fixes.join(matches, on="index").drop("index")


def pair_counts(vms: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
    """Count matched VMS fixes per (call sign, MMSI).

    Counts are of VMS fixes, not AIS fixes, so an MMSI reporting every
    10 s gains nothing over one reporting every minute.

    Args:
        vms: Fixes from ``cut_vms``.
        matches: Output of ``match_fixes``.

    Returns:
        pl.DataFrame: ``call_sign``, ``mmsi``, ``matched`` and
        ``matched_in_box``, joined to the call sign's ``vms_fixes`` and
        ``vms_fixes_in_box``; sorted by call sign, then most matches first.
    """
    totals = vms.group_by("call_sign").agg(
        vms_fixes=pl.len(), vms_fixes_in_box=pl.col("in_box").sum()
    )
    return (
        matches.group_by("call_sign", "mmsi")
        .agg(matched=pl.len(), matched_in_box=pl.col("in_box").sum())
        .join(totals, on="call_sign")
        .sort(
            ["call_sign", "matched_in_box", "matched"], descending=[False, True, True]
        )
    )


def step_match():
    """Match the VMS week to all-vessel HAIS and write matches and pairs."""
    vms = pl.read_parquet(OUT / "vms_week.parquet")
    ais = load_ais(str(RAW / "hais" / "*.parquet"))
    matches = match_fixes(vms, ais)
    pairs = pair_counts(vms, matches)
    matches.write_parquet(OUT / "matches.parquet", compression="zstd")
    pairs.write_parquet(OUT / "pairs.parquet", compression="zstd")
    matched_fixes = matches["message_id"].n_unique()
    print(f"AIS: {ais.height:,} reports, {ais['mmsi'].n_unique()} MMSIs")
    print(
        f"VMS fixes with any AIS match: {matched_fixes:,} of {vms.height:,} "
        f"({matched_fixes / vms.height:.0%}); in box only: "
        f"{matches.filter('in_box')['message_id'].n_unique():,} of "
        f"{vms['in_box'].sum():,}"
    )
    print(
        f"Pairs: {pairs.height:,} (call sign, MMSI) over "
        f"{pairs['call_sign'].n_unique()} call signs"
    )
    print(
        "Matches per VMS fix:",
        matches.group_by("message_id").len()["len"].value_counts().sort("len").rows(),
    )


def assign(vms: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """Decide which MMSI, if any, each eligible call sign belongs to.

    The rules are the plan's step 4, made exact before any assignment was
    computed. Only in-box fixes count, since HAIS has no data beyond the
    boxes. A call sign is eligible with at least ``MIN_FIXES`` in-box VMS
    fixes. Its leading MMSI is assigned when it matches at least
    ``MIN_FIXES`` of them, at least ``MIN_SHARE`` of them, and at least
    ``LEAD_RATIO`` times the runner-up's count. An MMSI that leads for
    more than one call sign makes all of them ambiguous.

    Args:
        vms: Fixes from ``cut_vms``.
        pairs: Output of ``pair_counts``.

    Returns:
        pl.DataFrame: One row per eligible call sign: ``vms_fixes_in_box``,
        ``mmsi`` and ``matched_in_box`` of the leader (null if none),
        ``runner_up_in_box`` (0 if none), ``share``, ``mmsis`` (how many
        MMSIs matched in the box), ``status`` (``assigned``, ``ambiguous``
        or ``unmatched``) and ``reason``.
    """
    eligible = (
        vms.filter("in_box")
        .group_by("call_sign")
        .agg(vms_fixes_in_box=pl.len())
        .filter(pl.col("vms_fixes_in_box") >= MIN_FIXES)
    )
    # Ties sort by MMSI so the leader is reproducible; an exact tie is
    # ambiguous anyway, because the leader cannot be twice the runner-up.
    ranked = (
        pairs.filter(pl.col("matched_in_box") > 0)
        .sort(["call_sign", "matched_in_box", "mmsi"], descending=[False, True, False])
        .group_by("call_sign", maintain_order=True)
        .agg(
            mmsi=pl.col("mmsi").first(),
            matched_in_box=pl.col("matched_in_box").first(),
            runner_up_in_box=pl.col("matched_in_box").get(1, null_on_oob=True),
            mmsis=pl.len(),
        )
    )
    table = eligible.join(ranked, on="call_sign", how="left").with_columns(
        pl.col("runner_up_in_box").fill_null(0),
        pl.col("mmsis").fill_null(0),
        share=pl.col("matched_in_box").fill_null(0) / pl.col("vms_fixes_in_box"),
    )
    leads_twice = pl.col("mmsi").is_not_null() & (pl.len().over("mmsi") > 1)
    status = (
        pl.when(pl.col("mmsi").is_null())
        .then(pl.lit("unmatched,no AIS match in box"))
        .when(pl.col("matched_in_box") < MIN_FIXES)
        .then(pl.lit(f"unmatched,leader under {MIN_FIXES} fixes"))
        .when(pl.col("share") < MIN_SHARE)
        .then(pl.lit(f"unmatched,leader under {MIN_SHARE:.0%} of fixes"))
        .when(pl.col("matched_in_box") < LEAD_RATIO * pl.col("runner_up_in_box"))
        .then(pl.lit(f"ambiguous,leader under {LEAD_RATIO:g}x runner-up"))
        .when(leads_twice)
        .then(pl.lit("ambiguous,MMSI leads another call sign"))
        .otherwise(pl.lit("assigned,"))
        .str.split_exact(",", 1)
        .struct.rename_fields(["status", "reason"])
        .alias("parts")
    )
    return (
        table.with_columns(status)
        .unnest("parts")
        .sort("call_sign")
        .select(
            "call_sign",
            "vms_fixes_in_box",
            "mmsi",
            "matched_in_box",
            "runner_up_in_box",
            "share",
            "mmsis",
            "status",
            "reason",
        )
    )


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a proportion.

    Better than the normal approximation at small samples and near 0 or 1,
    both of which apply to a few dozen call signs.

    Args:
        successes: Count of successes.
        trials: Count of trials; must be positive.
        z: Normal quantile; 1.96 gives 95%.

    Returns:
        tuple: (lower, upper) bounds as fractions.
    """
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = (
        z
        * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
        / (1 + z * z / trials)
    )
    return centre - half, centre + half


def step_assign():
    """Assign MMSIs to call signs and print the match rate."""
    vms = pl.read_parquet(OUT / "vms_week.parquet")
    table = assign(vms, pl.read_parquet(OUT / "pairs.parquet"))
    table.write_parquet(OUT / "assignments.parquet", compression="zstd")
    assigned = (table["status"] == "assigned").sum()
    low, high = wilson_interval(assigned, table.height)
    print(
        f"Eligible call signs: {table.height}; assigned {assigned} "
        f"({assigned / table.height:.0%}, 95% CI {low:.0%}-{high:.0%})"
    )
    print(table.group_by("status", "reason").len().sort("status", "reason"))
    with pl.Config(tbl_rows=-1):
        print(table.filter(pl.col("status") != "assigned").drop("call_sign"))


def compare_control(real: pl.DataFrame, control: pl.DataFrame) -> pl.DataFrame:
    """Set the time-shifted assignments beside the real ones.

    A control assignment to the same MMSI as the real run is most likely a
    vessel that stayed within 500 m of where it was 3 h earlier, not a
    stranger matched by chance; a different MMSI is a false match.

    Args:
        real: ``assign`` output for the unshifted match.
        control: ``assign`` output for the shifted match.

    Returns:
        pl.DataFrame: The control's assigned call signs, with
        ``control_mmsi``, ``real_mmsi``, ``real_status`` and ``same_mmsi``.
    """
    return (
        control.filter(pl.col("status") == "assigned")
        .select("call_sign", control_mmsi="mmsi")
        .join(
            real.select("call_sign", real_mmsi="mmsi", real_status="status"),
            on="call_sign",
        )
        .with_columns(
            same_mmsi=(pl.col("control_mmsi") == pl.col("real_mmsi")).fill_null(False)
        )
    )


def step_control():
    """Rerun match and assignment with VMS times shifted by ``SHIFT``."""
    vms = pl.read_parquet(OUT / "vms_week.parquet")
    ais = load_ais(str(RAW / "hais" / "*.parquet"))
    matches = match_fixes(vms, ais, shift=SHIFT)
    control = assign(vms, pair_counts(vms, matches))
    control.write_parquet(OUT / "control_assignments.parquet", compression="zstd")
    real = pl.read_parquet(OUT / "assignments.parquet")
    compared = compare_control(real, control)
    assigned = compared.height
    low, high = wilson_interval(assigned, control.height)
    late = vms.filter(
        pl.col("in_box"), pl.col("timestamp") + SHIFT > ais["timestamp"].max()
    ).height
    print(
        f"Control (+{SHIFT}): {matches['message_id'].n_unique():,} VMS fixes "
        f"match any MMSI; {late} in-box fixes fall past the AIS week"
    )
    print(
        f"Assigned {assigned} of {control.height} eligible "
        f"({assigned / control.height:.0%}, 95% CI {low:.0%}-{high:.0%}): "
        f"{compared['same_mmsi'].sum()} to the real run's MMSI, "
        f"{(~compared['same_mmsi']).sum()} to a different one"
    )
    print(control.group_by("status", "reason").len().sort("status", "reason"))


def delivery_mmsis(paths: str) -> set:
    """The MMSIs present in one HAIS delivery.

    Args:
        paths: A glob over the delivery's daily files.

    Returns:
        set: Distinct MMSIs as ints.
    """
    rows = duckdb.execute(
        f"select distinct mmsi from read_parquet('{paths}')"
    ).fetchall()
    return {int(row[0]) for row in rows}


def step_types():
    """Check assigned MMSIs against the fishing-only and cargo deliveries.

    HAIS has no ship type in its rows, but the order form filters by type,
    so presence in the fishing-only delivery says what HAIS thinks the
    vessel is.
    """
    real = pl.read_parquet(OUT / "assignments.parquet")
    assigned = set(real.filter(pl.col("status") == "assigned")["mmsi"].to_list())
    deliveries = {
        name: delivery_mmsis(str(RAW / name / "*.parquet"))
        for name in ("hais", "hais_fishing", "hais_cargo")
    }
    for name, mmsis in deliveries.items():
        print(
            f"{name}: {len(mmsis)} MMSIs; holds {len(assigned & mmsis)} "
            f"of {len(assigned)} assigned"
        )
    outside = assigned - deliveries["hais_fishing"]
    print(f"Assigned but not in the fishing delivery: {sorted(outside)}")
    print(
        f"Also in cargo: {sorted(outside & deliveries['hais_cargo'])}; "
        f"MID 257-259 (Norway): {sum(257_000_000 <= m < 260_000_000 for m in assigned)}"
        f" of {len(assigned)}"
    )


def main():
    """Run one spike step."""
    steps = {
        "vms": step_vms,
        "match": step_match,
        "assign": step_assign,
        "control": step_control,
        "types": step_types,
    }
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=list(steps))
    args = parser.parse_args()
    steps[args.step]()


if __name__ == "__main__":
    main()
