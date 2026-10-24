"""Tests for the committed US anchorages and distance to shore over the
real coastline."""

import hashlib

import numpy as np
import pytest

from vessels.core.distance import (
    MAX_SHORE_KM,
    MIN_VESSELS,
    PortDistance,
    ShoreDistance,
)
from vessels.core.geo import KM_PER_NM, LocalProjection
from vessels.io.blocks import BOX
from vessels.io.reference import (
    ANCHORAGES_US,
    REFERENCE_US,
    read_anchorages,
    read_land,
)

PROJECTION = LocalProjection((BOX[0] + BOX[2]) / 2, (BOX[1] + BOX[3]) / 2)
# Harbour waters the anchorages must cover (lon, lat).
HARBOURS = {
    "New Bedford": (-70.92, 41.63),
    "Gloucester": (-70.66, 42.61),
    "Portland": (-70.25, 43.65),
}


@pytest.fixture(name="anchorages", scope="module")
def fixture_anchorages():
    """Read the committed anchorages once.

    Returns:
        pl.DataFrame: The anchorage cells.
    """
    return read_anchorages()


@pytest.fixture(name="shore", scope="module")
def fixture_shore():
    """Distance to the committed coastline.

    Returns:
        ShoreDistance: Built once for the module.
    """
    return ShoreDistance(read_land(), PROJECTION)


def test_checksum_matches_committed_sums():
    """The committed anchorages are the ones ``SHA256SUMS`` records."""
    sums = dict(
        reversed(line.split("  "))
        for line in (REFERENCE_US / "SHA256SUMS").read_text().splitlines()
    )
    digest = hashlib.sha256(ANCHORAGES_US.read_bytes()).hexdigest()
    assert sums["anchorages.csv"] == digest


def test_every_anchorage_meets_the_rule(anchorages, shore):
    """Each cell has enough vessels and lies near shore, by the committed
    coastline."""
    assert (anchorages["vessels"] >= MIN_VESSELS).all()
    km = shore.km(anchorages["lon"], anchorages["lat"])
    np.testing.assert_allclose(km, anchorages["shore_km"], atol=1e-3)
    assert (km <= MAX_SHORE_KM).all()


def test_main_fishing_ports_are_anchorages(anchorages):
    """New Bedford, Gloucester and Portland harbours are within 3 km of an
    anchorage (GFW's port-visit distance)."""
    ports = PortDistance(anchorages["lon"], anchorages["lat"], PROJECTION)
    lon, lat = zip(*HARBOURS.values())
    assert (ports.km(lon, lat) < 3).all()


def test_offshore_banks_are_far_from_shore(shore):
    """Georges Bank lies beyond 50 nm, the Great South Channel between 20
    and 50 nm (east of Nantucket) and Jeffreys Ledge inside 20 nm."""
    km = shore.km([-67.5, -69.0, -70.2], [41.5, 41.3, 42.95])
    assert km[0] > 50 * KM_PER_NM
    assert 20 * KM_PER_NM < km[1] < 50 * KM_PER_NM
    assert 0 < km[2] < 20 * KM_PER_NM
