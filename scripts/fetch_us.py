"""Fetch US Northeast inputs: MarineCadastre 2024 AIS days and the closure
boundaries.

Optional: reviewers reproduce from the committed extracts. Run it with
``PYTHONPATH=src python scripts/fetch_us.py`` (the spike days by default)
``--blocks`` for every tuning and held-out day, or ``--day 2024-05-29``
for other days.

AIS: NOAA Office for Coastal Management, Marine Cadastre, from the US
Coast Guard NAIS; US Government material, not subject to copyright in the
United States, released for planning purposes only. Each national daily
GeoParquet file is about 230-310 MB, so the files stay in the git-ignored
``data/raw/marinecadastre/``. Closures: the GARFO Habitat Management Area
shapefile; 50 CFR 648.370-648.371 as in force on 2024-07-01, plus 648.370
on the dates its Great South Channel rules changed (2018-04-09 and
2020-06-18), from the eCFR point-in-time API (which serves only
compressed responses, so the XML is stored gzipped); the two Federal
Register notices that date and locate the 2024 exempted fishing permit in
the Great South Channel HMA; and the two ASMFC press releases that set the
northern shrimp moratorium for 2019-2024. Land: Natural Earth 10m land polygons (public
domain), for the land mask. Each download is recorded with its URL, retrieval
time and SHA-256 in ``SOURCES.json`` beside the files.
"""

import argparse
from datetime import date
from pathlib import Path

from vessels.io.blocks import BLOCKS
from vessels.io.fetch import fetch_sources

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
AIS_URL = "https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/ais-{day}.parquet"
# Spike (a): the 15th of one month per season, fixed before any download.
SPIKE_DAYS = [date(2024, m, 15) for m in (1, 4, 7, 10)]
AIS_PROVENANCE = {
    "licence": "US Government work (17 U.S.C. 105); planning purposes only",
    "attribution": "NOAA Office for Coastal Management, Marine Cadastre "
    "(US Coast Guard NAIS)",
}
CFR_URL = (
    "https://www.ecfr.gov/api/versioner/v1/full/{day}/title-50.xml"
    "?part=648&section={section}"
)
# Section and point-in-time date. 2024-07-01 is the study year; the other
# two are the eCFR versions in which the Great South Channel HMA rules
# changed (the 2018 clam dredge exemption and the 2020 exemption areas).
CFR_VERSIONS = [
    ("648.370", "2024-07-01"),
    ("648.371", "2024-07-01"),
    ("648.370", "2018-04-09"),
    ("648.370", "2020-06-18"),
]
FR_URL = "https://www.federalregister.gov/documents/full_text/text/{path}.txt"
ASMFC_URL = "https://asmfc.org/wp-content/uploads/{path}.pdf"


def cfr_source(section: str, day: str) -> tuple[str, dict]:
    """The eCFR point-in-time source for one section and date.

    The 2024-07-01 versions keep the names they were first recorded under,
    so existing manifests stay valid.

    Args:
        section: CFR section, such as ``648.370``.
        day: Point-in-time date, ISO format.

    Returns:
        tuple: Source name and its record (URL, file, headers, attribution).
    """
    stem = f"cfr_{section.replace('.', '_')}"
    name = stem if day == "2024-07-01" else f"{stem}_{day}"
    return name, {
        "url": CFR_URL.format(day=day, section=section),
        "file": f"{stem}_{day}.xml.gz",
        "headers": {"Accept-Encoding": "gzip"},
        "attribution": "eCFR, Office of the Federal Register",
    }


CLOSURES = {
    "hma": {
        "url": "https://www.fisheries.noaa.gov/s3/2020-04/"
        "habitat-management-areas-20180409-2-.zip",
        "file": "habitat-management-areas-20180409.zip",
        "attribution": "NOAA Fisheries GARFO (approximate; the CFR is legal)",
    },
    **dict(cfr_source(section, day) for section, day in CFR_VERSIONS),
    # The exempted fishing permit's application (area) and its extension
    # notice (issue and effective dates).
    **{
        f"fr_{number}": {
            "url": FR_URL.format(path=path),
            "file": f"fr_{number}.txt",
            "attribution": f"Federal Register {citation}",
        }
        for number, path, citation in [
            ("2024-05271", "2024/03/13/2024-05271", "89 FR 18376"),
            ("2025-11367", "2025/06/20/2025-11367", "90 FR 26274"),
        ]
    },
    **{
        f"asmfc_{name}": {
            "url": ASMFC_URL.format(path=path),
            "file": f"asmfc_{name}.pdf",
            "licence": "ASMFC press release; cited, not redistributed",
            "attribution": f"Atlantic States Marine Fisheries Commission, {title}",
        }
        for name, path, title in [
            (
                "pr18_40",
                "2025/10/pr40NorthernShrimpMoratorium",
                "news release 16 November 2018",
            ),
            (
                "pr21_32",
                "2025/09/pr32_NorthernShrimpMoratoriumExtended",
                "news release PR21-32, 17 December 2021",
            ),
        ]
    },
}
CLOSURE_PROVENANCE = {"licence": "US Government work (17 U.S.C. 105)"}
REFERENCE = {
    "land": {
        "url": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip",
        "file": "ne_10m_land.zip",
        "attribution": "Made with Natural Earth",
    },
}
REFERENCE_PROVENANCE = {"licence": "Public domain (Natural Earth terms of use)"}


def ais_sources(days: list[date]) -> dict:
    """The MarineCadastre daily files for some days.

    Args:
        days: 2024 days.

    Returns:
        dict: Source name to URL and file name.
    """
    return {
        f"ais_{day}": {
            "url": AIS_URL.format(day=day.isoformat()),
            "file": f"ais-{day.isoformat()}.parquet",
        }
        for day in days
    }


def main():
    """Download the AIS days and the closures not already present."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", type=date.fromisoformat, action="append")
    parser.add_argument(
        "--blocks", action="store_true", help="all tuning and held-out days"
    )
    parser.add_argument(
        "--force", action="store_true", help="download again even if verified"
    )
    args = parser.parse_args()
    fetch_sources(CLOSURES, RAW / "us_closures", CLOSURE_PROVENANCE, force=args.force)
    fetch_sources(REFERENCE, RAW / "reference", REFERENCE_PROVENANCE, force=args.force)
    days = sorted(
        set(args.day or [])
        | ({d for block in BLOCKS.values() for d in block} if args.blocks else set())
    )
    fetch_sources(
        ais_sources(days or SPIKE_DAYS),
        RAW / "marinecadastre",
        AIS_PROVENANCE,
        force=args.force,
    )


if __name__ == "__main__":
    main()
