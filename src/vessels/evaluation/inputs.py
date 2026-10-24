"""Load the committed reference layers a block's detectors need."""

from datetime import date
from pathlib import Path

import polars as pl

from vessels.core.distance import PortDistance, ShoreDistance
from vessels.core.geo import LocalProjection
from vessels.io.blocks import BOX
from vessels.io.marinecadastre import read_extract, read_reception
from vessels.io.reference import read_anchorages, read_closures, read_land
from vessels.stream.micro import Inputs

ROOT = Path(__file__).resolve().parents[3]
EXTRACT = ROOT / "data" / "extract" / "us"
GRID = ROOT / "results" / "coverage" / "tuning_coverage.parquet"


def heard_days(block: str) -> pl.DataFrame:
    """Cells and UTC days on which a block's reception grid heard traffic.

    Args:
        block: ``tuning`` or ``heldout``.

    Returns:
        pl.DataFrame: ``h3_cell`` and ``day``, one row per pair.
    """
    paths = sorted((EXTRACT / block / "reception").glob("*.parquet"))
    parts = [
        read_reception([path])
        .filter(pl.col("positions") > 0)
        .select("h3_cell")
        .unique()
        .with_columns(day=pl.lit(date.fromisoformat(path.stem)))
        .collect()
        for path in paths
    ]
    return pl.concat(parts).sort("day", "h3_cell")


def load(block: str) -> tuple[pl.DataFrame, Inputs]:
    """A block's positions and the inputs for every detector.

    The coverage grid is the tuning block's (``make coverage``) for both
    blocks: coverage is a tuned threshold, fixed before held-out scoring.

    Args:
        block: ``tuning`` or ``heldout``.

    Returns:
        tuple[pl.DataFrame, Inputs]: Positions in the extract schema, and
        the detector inputs.
    """
    positions = read_extract(sorted((EXTRACT / block).glob("*.parquet"))).collect()
    projection = LocalProjection((BOX[0] + BOX[2]) / 2, (BOX[1] + BOX[3]) / 2)
    anchorages = read_anchorages()
    inputs = Inputs(
        closures=read_closures(),
        grid=pl.read_parquet(GRID),
        ports=PortDistance(anchorages["lon"], anchorages["lat"], projection),
        heard=heard_days(block),
        shore=ShoreDistance(read_land(), projection),
    )
    return positions, inputs
