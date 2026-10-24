"""Tests for the coverage grid: the Wilson bound, daily totals, which cells
count as covered, and distance to the nearest uncovered cell."""

import h3
import polars as pl
import pytest

from vessels.core.coverage import (
    MAX_LONG_SHARE,
    Z95,
    coverage,
    cut_by_box,
    summarise,
    wilson_upper,
)

BOX = (-71.5, 40.0, -65.0, 45.0)
# A cell well inside the box, and one on its western edge.
INNER = h3.latlng_to_cell(42.5, -68.0, 6)
EDGE = h3.latlng_to_cell(42.5, -71.5, 6)


def grid_rows(cells, cls="A", moving=100, long=0, days=1):
    """Daily reception rows with the same counts for every cell and day.

    Args:
        cells: H3 cells.
        cls: AIS class.
        moving: Moving intervals per cell and day.
        long: Long intervals per cell and day.
        days: Days of rows per cell.

    Returns:
        pl.DataFrame: Rows in the reception grid schema.
    """
    n = len(cells) * days
    return pl.DataFrame(
        {
            "h3_cell": list(cells) * days,
            "transceiver": [cls] * n,
            "positions": [moving + 1] * n,
            "vessels": [2] * n,
            "moving_intervals": [moving] * n,
            "long_intervals": [long] * n,
        }
    )


def wilson(successes, trials):
    """Evaluate ``wilson_upper`` on plain numbers.

    Args:
        successes: Count of events.
        trials: Count of trials.

    Returns:
        float: The upper bound.
    """
    frame = pl.DataFrame({"k": [successes], "n": [trials]})
    return frame.select(wilson_upper(pl.col("k"), pl.col("n")))[0, 0]


def test_wilson_upper_matches_closed_form_and_needs_about_35_clean_intervals():
    """With no events the bound is z^2/(n+z^2), so 35 clean intervals pass 0.1."""
    for n in (1, 10, 34, 35, 1000):
        assert wilson(0, n) == pytest.approx(Z95**2 / (n + Z95**2))
    assert wilson(0, 34) > MAX_LONG_SHARE >= wilson(0, 35)


def test_wilson_upper_is_one_without_trials_and_bounded_by_one():
    """No trials gives no evidence; every event still stays at most 1."""
    assert wilson(0, 0) == 1.0
    assert wilson(5, 5) == pytest.approx(1.0)
    assert wilson(5, 50) == pytest.approx(0.2136, abs=1e-4)


def test_summarise_totals_days_and_leaves_share_null_without_intervals():
    """Daily rows add up per cell and class; no moving intervals means no share."""
    rows = pl.concat(
        [grid_rows([INNER], moving=10, long=1, days=3), grid_rows([EDGE], moving=0)]
    )
    summary = summarise(rows).rows(named=True)
    by_cell = {row["h3_cell"]: row for row in summary}
    inner = by_cell[INNER]
    assert (inner["days"], inner["moving_intervals"], inner["long_intervals"]) == (
        3,
        30,
        3,
    )
    assert inner["vessel_days"] == 6
    assert inner["long_share"] == pytest.approx(0.1)
    assert by_cell[EDGE]["long_share"] is None
    assert by_cell[EDGE]["long_share_upper"] == 1.0


def test_cut_by_box_flags_only_cells_reaching_outside():
    """A cell straddling the box edge is cut; one in the middle is not."""
    assert cut_by_box(EDGE, BOX)
    assert not cut_by_box(INNER, BOX)


def test_coverage_needs_clean_evidence_and_a_whole_cell():
    """Clean, well-sampled cells are covered; dropouts, thin data and the box edge are not."""
    noisy, thin, tiny = (h3.grid_ring(INNER, 3)[i] for i in (0, 5, 10))
    rows = pl.concat(
        [
            grid_rows([INNER, EDGE]),
            grid_rows([noisy], moving=100, long=20),
            grid_rows([thin], moving=20),
            grid_rows([tiny], moving=3, long=1),
        ]
    )
    result = coverage(summarise(rows), BOX)
    covered = dict(result.select("h3_cell", "covered").rows())
    assert covered == {
        INNER: True,
        EDGE: False,
        noisy: False,
        thin: False,
        tiny: False,
    }
    status = dict(result.select("h3_cell", "status").rows())
    assert status == {
        INNER: "covered",
        EDGE: "box edge",
        noisy: "dropouts",
        thin: "too few intervals",
        # A high share from three intervals is not evidence of dropouts.
        tiny: "too few intervals",
    }


def test_edge_km_grows_inward_and_classes_are_separate():
    """A cell inside a covered patch is further from the edge than its rim."""
    patch = h3.grid_disk(INNER, 2)
    rows = pl.concat([grid_rows(patch), grid_rows([INNER], cls="B")])
    result = coverage(summarise(rows), BOX)
    edge = {
        (c, cls): km
        for c, cls, km in result.select("h3_cell", "transceiver", "edge_km").rows()
    }
    rim = h3.grid_ring(INNER, 2)
    # Class A's centre cell is three rings from the nearest uncovered cell,
    # and the nearest cell of ring 3 is sqrt(7) neighbour spacings away;
    # Class B's lone cell has uncovered neighbours however Class A looks.
    assert edge[(INNER, "A")] > max(edge[(c, "A")] for c in rim)
    assert edge[(INNER, "A")] == pytest.approx(7**0.5 * edge[(INNER, "B")], rel=0.05)
    assert result.filter(~pl.col("covered"))["edge_km"].null_count() == (
        result.filter(~pl.col("covered")).height
    )


def test_coverage_keeps_rows_sorted_and_handles_an_empty_grid():
    """Output is sorted by cell and class, and an empty grid stays empty."""
    rows = grid_rows(h3.grid_disk(INNER, 1))
    result = coverage(summarise(rows), BOX)
    assert result["h3_cell"].to_list() == sorted(result["h3_cell"].to_list())
    empty = coverage(summarise(rows.clear()), BOX)
    assert empty.height == 0 and "edge_km" in empty.columns
