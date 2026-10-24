"""Build the committed US reference layers from raw downloads.

Owner only, though nothing in it is secret: run
``PYTHONPATH=src python scripts/fetch_us.py`` first, then
``PYTHONPATH=src python scripts/build_us_reference.py``. It writes, under
``data/reference/us/``:

- ``land.geojson``: Natural Earth 10m land clipped to the study box plus a
  margin, so the clip edge does not act as a coastline for the land
  mask's inland buffer. Coordinates are rounded to 1e-6 degrees (about
  0.1 m) to keep the file small and stable;
- ``closures.geojson``: the dated geofence table (closures, exemptions
  and suspensions; see ``vessels.core.geofence``) for every Habitat
  Management Area and Dedicated Habitat Research Area that bans gear in
  the study box, with the same rounding;
- ``SHA256SUMS`` for every layer.

Each date and boundary is read from, or checked against, the source text,
so a source that changes under the same URL fails the build.
"""

import json
import re
from datetime import date
from pathlib import Path

import geopandas as gpd
import shapely

from vessels.core.geofence import Rule, check_table
from vessels.io.blocks import BOX
from vessels.io.reference import REFERENCE_US, write_sums
from vessels.io.us_closures import cfr_polygon, cfr_table, cfr_text

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "reference"
CLOSURE_RAW = ROOT / "data" / "raw" / "us_closures"
HMA_SHAPEFILE = (
    "zip://"
    + str(CLOSURE_RAW / "habitat-management-areas-20180409.zip")
    + "!Habitat_Management_Areas/Habitat_Management_Areas.shp"
)
# Degrees added on every side of the box; far more than any land buffer.
MARGIN = 0.5
PRECISION = 1e-6
TIMEZONE = "America/New_York"
# Omnibus Essential Fish Habitat Amendment 2 final rule, effective on
# publication.
OHA2 = ("2018-04-09", "83 FR 15240")
CLAM_FRAMEWORK = ("2020-06-18", "85 FR 29870")


def land_geojson(zip_path: Path) -> dict:
    """Clip Natural Earth land to the padded study box.

    Args:
        zip_path: The downloaded ``ne_10m_land.zip``.

    Returns:
        dict: A GeoJSON FeatureCollection with one MultiPolygon feature.
    """
    min_lon, min_lat, max_lon, max_lat = BOX
    padded = (min_lon - MARGIN, min_lat - MARGIN, max_lon + MARGIN, max_lat + MARGIN)
    land = gpd.read_file(f"zip://{zip_path}").clip(padded)
    geom = shapely.set_precision(shapely.union_all(land.geometry.values), PRECISION)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"source": "Natural Earth 10m land"},
                "geometry": json.loads(shapely.to_geojson(geom)),
            }
        ],
    }


def require(text: str, phrase: str, source: str) -> None:
    """Fail unless a source still says what the table relies on.

    Whitespace is collapsed on both sides, so line wrapping does not
    matter.

    Args:
        text: The source's text.
        phrase: The wording the table depends on.
        source: Name for the error message.

    Raises:
        ValueError: If the phrase is missing.
    """
    if " ".join(phrase.split()) not in " ".join(text.split()):
        raise ValueError(f"{source} no longer says {phrase!r}")


def efp_area(text: str) -> shapely.Polygon:
    """Read the exempted fishing permit's area from its Federal Register
    notice.

    The notice lists four decimal-degree corners, not in ring order (and
    not a rectangle: the south edge is shorter), so the area is their
    convex hull, which must keep all four as vertices.

    Args:
        text: The notice's text.

    Returns:
        Polygon: The quadrilateral in longitude and latitude.

    Raises:
        ValueError: If the notice does not list four corners of one convex
            quadrilateral.
    """
    corners = re.findall(r"([\d.]+)\[deg\] N\.+\s+([\d.]+)\[deg\] W", text)
    hull = shapely.MultiPoint(
        [(-float(lon), float(lat)) for lat, lon in corners]
    ).convex_hull
    if len(corners) != 4 or len(shapely.get_coordinates(hull)) != 5:
        raise ValueError(f"EFP area: unexpected corners {corners}")
    return shapely.orient_polygons(hull)


def closure_rows() -> list[dict]:
    """Assemble every row of the geofence table from the raw sources.

    Returns:
        list[dict]: Rule fields plus the documenting properties (``name``,
        ``citation``, ``source_type``, ``geometry_source``, ``status``,
        ``note``), in table order.

    Raises:
        ValueError: If a source no longer says what a row relies on.
    """
    cfr24 = cfr_text(CLOSURE_RAW / "cfr_648_370_2024-07-01.xml.gz")
    cfr18 = cfr_text(CLOSURE_RAW / "cfr_648_370_2018-04-09.xml.gz")
    cfr20 = cfr_text(CLOSURE_RAW / "cfr_648_370_2020-06-18.xml.gz")
    dhra = cfr_text(CLOSURE_RAW / "cfr_648_371_2024-07-01.xml.gz")
    efp_application = (CLOSURE_RAW / "fr_2024-05271.txt").read_text(encoding="utf-8")
    efp_extension = (CLOSURE_RAW / "fr_2025-11367.txt").read_text(encoding="utf-8")
    shapefile = gpd.read_file(HMA_SHAPEFILE).set_index("CFRPARA").geometry

    require(cfr18, "effective until April 9, 2019", "648.370 (2018)")
    require(cfr24, "open from May 1-October 31", "648.370 (2024)")
    require(cfr24, "McBlair Dredge Exemption Area is open year-round", "648.370")
    require(cfr24, "Fishing Rip Dredge Exemption Area is open year-round", "648.370")
    require(
        dhra,
        "bottom-tending mobile gear, sink gillnet gear, or demersal "
        "longline gear are prohibited from fishing in the Stellwagen DHRA",
        "648.371",
    )
    require(
        efp_extension,
        "On May 24, 2024, NMFS issued the EFP effective "
        "August 1, 2024, through July 31, 2025",
        "90 FR 26274",
    )
    require(
        efp_extension, "NMFS issued a revised EFP on November 6, 2024", "90 FR 26274"
    )
    # The 2018 and 2020 versions must draw the same HMA as 2024, so one
    # polygon serves every date.
    gsc = cfr_polygon(cfr24, "GSCH")
    for text in (cfr18, cfr20):
        if not cfr_polygon(text, "GSCH").equals(gsc):
            raise ValueError("Great South Channel HMA changed between versions")

    def hma(area_id, name, para, geometry, source="cfr"):
        """A 648.370 closure in force from Amendment 2 onwards.

        Args:
            area_id: Row identifier.
            name: Human-readable name.
            para: Paragraph of 648.370, such as ``h(1)``.
            geometry: The closure polygon.
            source: Where the geometry came from.

        Returns:
            dict: The row.
        """
        return dict(
            area_id=area_id,
            name=name,
            kind="closure",
            parent=None,
            gear=["mobile_bottom"],
            valid_from=OHA2[0],
            valid_to=None,
            season=None,
            geometry=geometry,
            citation=f"50 CFR 648.370({para}); {OHA2[1]}",
            source_type="cfr",
            geometry_source=source,
            status="observed",
            note=None,
        )

    def dredge_area(area_id, name, para, heading, season=None):
        """A 2020 Great South Channel dredge exemption area.

        Args:
            area_id: Row identifier.
            name: Human-readable name.
            para: Sub-paragraph of 648.370(h)(2), such as ``ii``.
            heading: Text just before the area's point table.
            season: Inclusive month-day window, or None for all year.

        Returns:
            dict: The row.
        """
        return dict(
            area_id=area_id,
            name=name,
            kind="exemption",
            parent="gsc_hma",
            gear=["hydraulic_clam_dredge", "mussel_dredge"],
            valid_from=CLAM_FRAMEWORK[0],
            valid_to=None,
            season=season,
            geometry=cfr_table(cfr24, heading),
            citation=f"50 CFR 648.370(h)(2)({para}); {CLAM_FRAMEWORK[1]}",
            source_type="cfr",
            geometry_source="cfr",
            status="observed",
            note="Surfclam permit, Letter of Authorization, VMS and trip "
            "declaration required; gear is not in AIS.",
        )

    shrimp_area = cfr_polygon(cfr24, "SEA")
    rows = [
        hma(
            "eastern_maine_hma",
            "Eastern Maine HMA",
            "a",
            shapefile["648.370(a)"],
            "garfo_shapefile",
        ),
        hma("jeffreys_bank_hma", "Jeffreys Bank HMA", "b", cfr_polygon(cfr24, "JBH")),
        hma("cashes_ledge_hma", "Cashes Ledge HMA", "c", cfr_polygon(cfr24, "CLH")),
        hma(
            "fippennies_ledge_hma",
            "Fippennies Ledge HMA",
            "d",
            cfr_polygon(cfr24, "FLH"),
        ),
        hma("ammen_rock_hma", "Ammen Rock HMA", "e", cfr_polygon(cfr24, "ARH")),
        hma(
            "wgom_hma", "Western Gulf of Maine HMA", "f(1)", cfr_polygon(cfr24, "WGMH")
        ),
        hma(
            "closed_area_ii_habitat",
            "Closed Area II Habitat Closure Area",
            "g",
            shapefile["648.370(g)"],
            "garfo_shapefile",
        ),
        hma("gsc_hma", "Great South Channel HMA", "h(1)", gsc),
        dict(
            area_id="stellwagen_dhra",
            name="Stellwagen DHRA",
            kind="closure",
            parent=None,
            gear=["mobile_bottom", "sink_gillnet", "demersal_longline"],
            valid_from=OHA2[0],
            valid_to=None,
            season=None,
            geometry=cfr_polygon(dhra, "SDHRA"),
            citation=f"50 CFR 648.371(b); {OHA2[1]}",
            source_type="cfr",
            geometry_source="cfr",
            status="observed",
            note=None,
        ),
        dict(
            area_id="georges_bank_dhra",
            name="Georges Bank DHRA",
            kind="closure",
            parent=None,
            gear=["mobile_bottom"],
            valid_from=OHA2[0],
            valid_to=None,
            season=None,
            geometry=cfr_polygon(dhra, "GBDHRA"),
            citation=f"50 CFR 648.371(c); {OHA2[1]}",
            source_type="cfr",
            geometry_source="cfr",
            status="observed",
            note=None,
        ),
        dict(
            area_id="wgom_shrimp_ea",
            name="Western Gulf of Maine Shrimp " "Exemption Area",
            kind="exemption",
            parent="wgom_hma",
            gear=["shrimp_trawl"],
            valid_from=OHA2[0],
            valid_to=None,
            season=None,
            geometry=shrimp_area,
            citation=f"50 CFR 648.370(f)(2); {OHA2[1]}",
            source_type="cfr",
            geometry_source="cfr",
            status="observed",
            note="Small Mesh Northern Shrimp Fishery Exemption vessels only.",
        ),
        *[
            dict(
                area_id=f"wgom_shrimp_moratorium_{start[:4]}",
                name=f"Northern shrimp moratorium {start[:4]}-{int(end[:4]) - 1}",
                kind="suspension",
                parent="wgom_shrimp_ea",
                gear=["shrimp_trawl"],
                valid_from=start,
                valid_to=end,
                season=None,
                geometry=shrimp_area,
                citation=citation,
                source_type="asmfc",
                geometry_source="parent",
                status="provisional",
                note="ASMFC moratorium, enforced through state landing rules, "
                "not the CFR; dates from the press release, not the addendum.",
            )
            for start, end, citation in [
                (
                    "2019-01-01",
                    "2022-01-01",
                    "ASMFC news release, 16 November 2018 (moratorium through 2021)",
                ),
                (
                    "2022-01-01",
                    "2025-01-01",
                    "ASMFC news release PR21-32, 17 December 2021 "
                    "(moratorium through 2024)",
                ),
            ]
        ],
        dict(
            area_id="gsc_clam_2018",
            name="Great South Channel hydraulic clam " "dredge exemption (2018)",
            kind="exemption",
            parent="gsc_hma",
            gear=["hydraulic_clam_dredge"],
            valid_from=OHA2[0],
            # "Effective until April 9, 2019, after which" no clam dredging:
            # 9 April is the last day.
            valid_to="2019-04-10",
            season=None,
            geometry=gsc.difference(
                cfr_table(cfr18, "does not apply in the area defined")
            ),
            citation=f"50 CFR 648.370(h)(2) as of 2018-04-09; {OHA2[1]}",
            source_type="cfr",
            geometry_source="cfr",
            status="observed",
            note="Surfclam and ocean quahog permits. The HMA minus the "
            "corner GARFO's shapefile calls the MBTG Closure.",
        ),
        dredge_area(
            "gsc_mcblair",
            "McBlair Dredge Exemption Area",
            "ii",
            "McBlair Dredge Exemption Area is defined",
        ),
        dredge_area(
            "gsc_old_south",
            "Old South Dredge Exemption Area",
            "iii",
            "Old South Dredge Exemption Area is defined",
            ["05-01", "10-31"],
        ),
        dredge_area(
            "gsc_fishing_rip",
            "Fishing Rip Dredge Exemption Area",
            "iv",
            "Fishing Rip Dredge Exemption Area is defined",
        ),
        dict(
            area_id="gsc_efp_davis_bank_east",
            name="CFF exempted fishing " "permit, Davis Bank East",
            kind="exemption",
            parent="gsc_hma",
            gear=["hydraulic_clam_dredge"],
            valid_from="2024-08-01",
            valid_to="2024-11-06",
            season=None,
            geometry=efp_area(efp_application),
            citation="EFP DA23-073; 89 FR 18376; 90 FR 26274",
            source_type="efp",
            geometry_source="fr_notice",
            status="provisional",
            note="Two active vessels. Ends at the revised permit of "
            "2024-11-06, whose area was not published.",
        ),
    ]
    return rows


def closures_geojson(rows: list[dict]) -> dict:
    """Validate the rows as rules and write them as GeoJSON.

    Args:
        rows: ``closure_rows`` output.

    Returns:
        dict: A GeoJSON FeatureCollection, one feature per row, with
        coordinates snapped to ``PRECISION``.

    Raises:
        ValueError: If a row is not a valid rule, the table does not fit
            together, or an exemption or suspension leaves its parent.
    """
    features, rules = [], {}
    for row in rows:
        props = {**row, "timezone": TIMEZONE}
        geometry = shapely.set_precision(props.pop("geometry"), PRECISION)
        rule = Rule(
            area_id=row["area_id"],
            kind=row["kind"],
            gear=tuple(row["gear"]),
            valid_from=_day(row["valid_from"]),
            valid_to=_day(row["valid_to"]),
            geometry=geometry,
            parent=row["parent"],
            season=tuple(row["season"]) if row["season"] else None,
            timezone=TIMEZONE,
        )
        rules[rule.area_id] = rule
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": json.loads(shapely.to_geojson(geometry)),
            }
        )
    check_table(list(rules.values()))
    for rule in rules.values():
        # A snapped edge may shift by half the precision either way.
        if rule.parent and not rules[rule.parent].geometry.buffer(PRECISION).covers(
            rule.geometry
        ):
            raise ValueError(f"{rule.area_id} leaves its parent {rule.parent}")
    return {"type": "FeatureCollection", "features": features}


def _day(value):
    """Parse an ISO date, keeping None.

    Args:
        value: ISO date string or None.

    Returns:
        date | None: The date.
    """
    return None if value is None else date.fromisoformat(value)


def write_json(path: Path, data: dict) -> None:
    """Write JSON with a trailing newline, the same bytes every run.

    Args:
        path: Destination.
        data: The object.
    """
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def main():
    """Write every reference layer and its checksums."""
    REFERENCE_US.mkdir(parents=True, exist_ok=True)
    land = REFERENCE_US / "land.geojson"
    write_json(land, land_geojson(RAW / "ne_10m_land.zip"))
    closures = REFERENCE_US / "closures.geojson"
    write_json(closures, closures_geojson(closure_rows()))
    write_sums(REFERENCE_US)
    for path in (land, closures):
        print(f"{path}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
