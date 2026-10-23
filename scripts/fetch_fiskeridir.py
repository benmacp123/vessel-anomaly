"""Fetch the Norwegian 2024 VMS and ERS open data and the coral-reef
closures from Fiskeridirektoratet.

Optional: only the identity-match spike and the Norwegian extracts read
these files, and reviewers reproduce from the committed extracts. Run it
with ``PYTHONPATH=src python scripts/fetch_fiskeridir.py``.

All three are published under the Norwegian Licence for Open Government
Data (NLOD 2.0); credit Fiskeridirektoratet. VMS and ERS name vessels and
call signs, so everything stays in the git-ignored ``data/raw/fiskeridir/``.
The closures come from a live WFS, so they are the layer as served on the
day of the fetch, not as it stood in 2024. Each download is hashed as it
streams, checked against the server's length, and recorded with its URL,
retrieval time and SHA-256 in ``SOURCES.json`` beside the files.
"""

import argparse
from pathlib import Path

from vessels.io.fetch import fetch_sources

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "fiskeridir"

# Archive URLs as listed on the Åpne data pages for VMS and ERS
# (fiskeridir.no, checked 2026-10-23).
SOURCES = {
    "vms": {
        "url": "https://register.fiskeridir.no/vms-ers/2024-VMS.csv.zip",
        "file": "2024-VMS.csv.zip",
    },
    "ers": {
        "url": "https://register.fiskeridir.no/vms-ers/ERS/"
        "elektronisk-rapportering-ers-2024.zip",
        "file": "elektronisk-rapportering-ers-2024.zip",
    },
    # Coral-reef closures (Korallrev forbudsområder) as GML 3.2 in
    # EPSG:4326, whose axis order is latitude, longitude.
    "coral": {
        "url": "https://wfs.geonorge.no/skwms1/wfs.korallrevforbudsomrader"
        "?service=WFS&version=2.0.0&request=GetFeature&typeNames=app:Korallrev"
        "&namespaces=xmlns(app,http://skjema.geonorge.no/SOSI/"
        "produktspesifikasjon/KorallrevForbud/20170601)"
        "&srsName=urn:ogc:def:crs:EPSG::4326",
        "file": "korallrev.gml",
    },
}
PROVENANCE = {"licence": "NLOD-2.0", "attribution": "Fiskeridirektoratet"}


def main():
    """Download each source not already present and verified."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(SOURCES), action="append")
    parser.add_argument(
        "--force", action="store_true", help="download again even if verified"
    )
    parser.add_argument("--out", type=Path, default=OUT, help="output directory")
    args = parser.parse_args()
    chosen = {name: SOURCES[name] for name in args.only or sorted(SOURCES)}
    fetch_sources(chosen, args.out, PROVENANCE, force=args.force)


if __name__ == "__main__":
    main()
