"""Tests for the coverage map: it draws every status for every class."""

import h3
import polars as pl
from shapely.geometry import box

from vessels.core.coverage import coverage, summarise
from vessels.evaluation.maps import plot_coverage

BOX = (-71.5, 40.0, -65.0, 45.0)


def test_plot_coverage_writes_a_png_with_every_status(tmp_path):
    """A grid holding all four statuses and two classes renders to a PNG."""
    centre = h3.latlng_to_cell(42.5, -68.0, 6)
    cells = [centre, h3.latlng_to_cell(42.5, -71.5, 6)] + h3.grid_ring(centre, 3)[:2]
    grid = pl.DataFrame(
        {
            "h3_cell": cells * 2,
            "transceiver": ["A"] * 4 + ["B"] * 4,
            "positions": [10] * 8,
            "vessels": [1] * 8,
            "moving_intervals": [100, 100, 100, 5] * 2,
            "long_intervals": [0, 0, 50, 0] * 2,
        }
    )
    result = coverage(summarise(grid), BOX)
    assert set(result["status"]) == {
        "covered",
        "box edge",
        "dropouts",
        "too few intervals",
    }
    path = tmp_path / "map.png"
    plot_coverage(result, box(-71, 42, -70, 43), BOX, path, "Title", "Subtitle")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
