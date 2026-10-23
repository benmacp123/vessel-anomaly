"""Fetch the Norwegian 2024 VMS and ERS open data from Fiskeridirektoratet.

Optional: only the identity-match spike and the Norwegian extracts read
these files, and reviewers reproduce from the committed extracts. Run it
with ``python scripts/fetch_fiskeridir.py``.

Both archives are published under the Norwegian Licence for Open
Government Data (NLOD 2.0); credit Fiskeridirektoratet. They name vessels
and call signs, so they stay in the git-ignored ``data/raw/fiskeridir/``.
Each download is hashed as it streams, checked against the server's
length, and recorded with its URL, retrieval time and SHA-256 in
``SOURCES.json`` beside the files.
"""

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "fiskeridir"
USER_AGENT = "vessel-anomaly portfolio project (open data fetch)"
CHUNK = 1 << 20

# Archive URLs as listed on the Åpne data pages for VMS and ERS
# (fiskeridir.no, checked 2026-10-23).
SOURCES = {
    "vms": "https://register.fiskeridir.no/vms-ers/2024-VMS.csv.zip",
    "ers": "https://register.fiskeridir.no/vms-ers/ERS/elektronisk-rapportering-ers-2024.zip",
}


def sha256_of(path: Path) -> str:
    """Hash a file on disk.

    Args:
        path: File to hash.

    Returns:
        str: Hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> dict:
    """Stream a URL to a file, hashing it on the way.

    The data go to a ``.part`` file first and are renamed only once the
    byte count matches the server's ``Content-Length``, so an interrupted
    run never leaves a truncated archive under the final name.

    Args:
        url: Source URL.
        path: Destination file.

    Returns:
        dict: The provenance record: URL, file name, retrieval time (UTC),
        size, SHA-256 and the server's ``Last-Modified`` header.

    Raises:
        OSError: If the download is shorter or longer than announced.
        urllib.error.URLError: If the request fails.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=120) as response:
        expected = response.headers.get("Content-Length")
        modified = response.headers.get("Last-Modified")
        with partial.open("wb") as handle:
            while chunk := response.read(CHUNK):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    if expected is not None and size != int(expected):
        partial.unlink()
        raise OSError(f"{url}: got {size:,} bytes, expected {int(expected):,}")
    partial.replace(path)
    return {
        "url": url,
        "file": path.name,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "last_modified": modified,
        "licence": "NLOD-2.0",
        "attribution": "Fiskeridirektoratet",
    }


def load_sources(path: Path) -> dict:
    """Read the provenance records written by earlier runs.

    Args:
        path: The ``SOURCES.json`` file.

    Returns:
        dict: Records keyed by source name; empty if the file is missing.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    """Download each archive not already present and verified."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(SOURCES), action="append")
    parser.add_argument(
        "--force", action="store_true", help="download again even if verified"
    )
    parser.add_argument("--out", type=Path, default=OUT, help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "SOURCES.json"
    records = load_sources(manifest)
    for name in args.only or sorted(SOURCES):
        url = SOURCES[name]
        path = args.out / url.rsplit("/", 1)[1]
        record = records.get(name)
        # Skip a file that is on disk and still matches its recorded hash;
        # anything else (missing, changed or never recorded) is fetched.
        if (
            not args.force
            and record
            and path.exists()
            and sha256_of(path) == record["sha256"]
        ):
            print(f"{name}: {path.name} present, SHA-256 matches; skipped")
            continue
        print(f"{name}: downloading {url}")
        records[name] = download(url, path)
        manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(
            f"{name}: {records[name]['bytes']:,} bytes, "
            f"sha256 {records[name]['sha256']}"
        )


if __name__ == "__main__":
    main()
