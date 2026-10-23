"""First spike: match Norwegian VMS call signs to AIS MMSIs.

Local only: it reads the git-ignored raw files and writes its working
data to ``data/raw/spike/``. The pass and fail criteria are fixed in the
project plan before running. Steps are subcommands, run in order:

    PYTHONPATH=src python scripts/spike_identity.py vms

``vms`` cuts the spike week out of the 2024 VMS archive, keeps fixes
inside the five HAIS order boxes plus a buffer, and drops repeated
reports of the same vessel in the same minute.
"""

import argparse
import math
from datetime import date, timedelta
from pathlib import Path

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
        pl.DataFrame: The input columns plus ``box``, sorted by call sign
        and time.
    """
    return (
        vms.with_columns(box_label(BOXES, BUFFER_KM))
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


def main():
    """Run one spike step."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["vms"])
    args = parser.parse_args()
    {"vms": step_vms}[args.step]()


if __name__ == "__main__":
    main()
