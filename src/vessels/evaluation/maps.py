"""Maps of H3 cell layers over the study box, for checking by eye.

Colours follow one scheme across figures: a single blue ramp for a
magnitude (light near zero, dark for more), orange for cells with a
problem, greys for no evidence and for the box edge.
"""

from pathlib import Path

import h3
import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from shapely.geometry.base import BaseGeometry

SURFACE = "#fcfcfb"
LAND = "#e4e2dd"
INK = "#3d3d3a"
DROPOUTS = "#eb6834"
NO_EVIDENCE = "#d6d5d0"
BOX_EDGE = "#9a9994"
# Blue ramp steps 250, 400, 550 and 700: covered cells by distance to
# the nearest uncovered cell, one step per band.
EDGE_BANDS = (
    (0.0, 10.0, "#86b6ef"),
    (10.0, 20.0, "#3987e5"),
    (20.0, 40.0, "#1c5cab"),
    (40.0, np.inf, "#0d366b"),
)


def _polygons(cells: list[str]) -> list[list[tuple[float, float]]]:
    """Cell outlines as longitude-latitude vertex lists.

    Args:
        cells: H3 cells.

    Returns:
        list[list[tuple[float, float]]]: One vertex list per cell.
    """
    return [[(lon, lat) for lat, lon in h3.cell_to_boundary(c)] for c in cells]


def _draw_land(ax, land: BaseGeometry) -> None:
    """Fill land polygons above the cells.

    Args:
        ax: Matplotlib axes.
        land: Land geometry in degrees.
    """
    for polygon in getattr(land, "geoms", [land]):
        x, y = polygon.exterior.xy
        ax.fill(x, y, color=LAND, edgecolor="#a8a6a0", linewidth=0.3, zorder=2)


def _edge_colour(km: float) -> str:
    """Ramp colour for a covered cell's distance to the edge.

    Args:
        km: Distance to the nearest uncovered cell.

    Returns:
        str: Hex colour.
    """
    return next(colour for low, high, colour in EDGE_BANDS if low <= km < high)


def plot_coverage(
    result: pl.DataFrame,
    land: BaseGeometry,
    box: tuple[float, float, float, float],
    path: Path,
    title: str,
    subtitle: str,
) -> None:
    """Map coverage status per AIS class, one panel per class.

    Args:
        result: ``vessels.core.coverage.coverage`` output.
        land: Land geometry in degrees.
        box: Study box ``(min_lon, min_lat, max_lon, max_lat)``.
        path: PNG to write.
        title: Figure title.
        subtitle: One line under the title: the rule and how to read it.
    """
    classes = sorted(result["transceiver"].drop_nulls().unique())
    fig, axes = plt.subplots(
        1, len(classes), figsize=(7 * len(classes), 7.4), facecolor=SURFACE
    )
    for ax, cls in zip(np.atleast_1d(axes), classes):
        rows = result.filter(pl.col("transceiver") == cls)
        colours = [
            {
                "dropouts": DROPOUTS,
                "too few intervals": NO_EVIDENCE,
                "box edge": BOX_EDGE,
            }.get(status)
            or _edge_colour(km)
            for status, km in zip(rows["status"], rows["edge_km"])
        ]
        ax.add_collection(
            PolyCollection(
                _polygons(rows["h3_cell"].to_list()),
                facecolors=colours,
                edgecolors="none",
                zorder=1,
            )
        )
        _draw_land(ax, land)
        min_lon, min_lat, max_lon, max_lat = box
        ax.set_xlim(min_lon - 0.1, max_lon + 0.1)
        ax.set_ylim(min_lat - 0.1, max_lat + 0.1)
        ax.set_aspect(1 / np.cos(np.radians((min_lat + max_lat) / 2)))
        ax.set_facecolor(SURFACE)
        covered = rows.filter(pl.col("covered")).height
        ax.set_title(
            f"Class {cls}: {covered:,} of {rows.height:,} cells with reception "
            "covered",
            color=INK,
            fontsize=11,
            loc="left",
        )
        ax.tick_params(colors=INK, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#c9c7c1")
    handles = [
        Patch(color=colour, label=f"Covered, {low:g}–{high:g} km from edge")
        for low, high, colour in EDGE_BANDS[:-1]
    ] + [
        Patch(color=EDGE_BANDS[-1][2], label="Covered, 40 km or more from edge"),
        Patch(color=DROPOUTS, label="Dropouts (long-interval share too high)"),
        Patch(color=NO_EVIDENCE, label="Too few moving intervals to tell"),
        Patch(color=BOX_EDGE, label="Cut by the study box"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    fig.suptitle(title, color=INK, fontsize=13, x=0.02, ha="left")
    fig.text(
        0.02,
        0.935,
        subtitle,
        color="#6b6a66",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=0.12, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
