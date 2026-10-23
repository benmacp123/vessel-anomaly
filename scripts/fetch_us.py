"""Fetch US Northeast inputs: MarineCadastre 2024 AIS days and the closure
boundaries.

Optional: reviewers reproduce from the committed extracts. Run it with
``PYTHONPATH=src python scripts/fetch_us.py`` (the spike days by default)
or ``--day 2024-05-29`` for other days.

AIS: NOAA Office for Coastal Management, Marine Cadastre, from the US
Coast Guard NAIS; US Government material, not subject to copyright in the
United States, released for planning purposes only. Each national daily
GeoParquet file is about 230-310 MB, so the files stay in the git-ignored
``data/raw/marinecadastre/``. Closures: the GARFO Habitat Management Area
shapefile and 50 CFR 648.370-648.371 as in force on 2024-07-01, from the
eCFR point-in-time API (which serves only compressed responses, so the
XML is stored gzipped). Each download is recorded with its URL, retrieval
time and SHA-256 in ``SOURCES.json`` beside the files.
"""

import argparse
from datetime import date
from pathlib import Path

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
    "https://www.ecfr.gov/api/versioner/v1/full/2024-07-01/title-50.xml"
    "?part=648&section={section}"
)
CLOSURES = {
    "hma": {
        "url": "https://www.fisheries.noaa.gov/s3/2020-04/"
        "habitat-management-areas-20180409-2-.zip",
        "file": "habitat-management-areas-20180409.zip",
        "attribution": "NOAA Fisheries GARFO (approximate; the CFR is legal)",
    },
    **{
        f"cfr_{section.replace('.', '_')}": {
            "url": CFR_URL.format(section=section),
            "file": f"cfr_{section.replace('.', '_')}_2024-07-01.xml.gz",
            "headers": {"Accept-Encoding": "gzip"},
            "attribution": "eCFR, Office of the Federal Register",
        }
        for section in ("648.370", "648.371")
    },
}
CLOSURE_PROVENANCE = {"licence": "US Government work (17 U.S.C. 105)"}


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
        "--force", action="store_true", help="download again even if verified"
    )
    args = parser.parse_args()
    fetch_sources(CLOSURES, RAW / "us_closures", CLOSURE_PROVENANCE, force=args.force)
    fetch_sources(
        ais_sources(args.day or SPIKE_DAYS),
        RAW / "marinecadastre",
        AIS_PROVENANCE,
        force=args.force,
    )


if __name__ == "__main__":
    main()
