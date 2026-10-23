"""Hashed downloads with a provenance manifest, shared by the fetch scripts.

Each source is streamed to disk, hashed on the way, checked against the
server's length and recorded in a ``SOURCES.json`` beside the files, so a
rerun can skip anything already present and unchanged.
"""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = "vessel-anomaly portfolio project (open data fetch)"
CHUNK = 1 << 20


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


def download(url: str, path: Path, headers: dict | None = None) -> dict:
    """Stream a URL to a file, hashing it on the way.

    The data go to a ``.part`` file first and are renamed only once the
    byte count matches the server's ``Content-Length``, so an interrupted
    run never leaves a truncated file under the final name. The bytes are
    stored as sent: a gzip-encoded response stays gzip on disk.

    Args:
        url: Source URL.
        path: Destination file.
        headers: Extra request headers.

    Returns:
        dict: The provenance record: URL, file name, retrieval time (UTC),
        size, SHA-256 and the server's ``Last-Modified`` header.

    Raises:
        OSError: If the download is shorter or longer than announced.
        urllib.error.URLError: If the request fails.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **(headers or {})}
    )
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


def fetch_sources(sources: dict, out: Path, extra: dict, force: bool = False) -> dict:
    """Download each source not already present and verified.

    The manifest is rewritten after every download, so an interrupted run
    keeps the records of what it finished.

    Args:
        sources: Name to a dict with ``url``, ``file`` and optionally
            ``headers`` and per-source fields to record (such as
            ``licence``).
        out: Directory for the files and ``SOURCES.json``.
        extra: Fields added to every record, overridden per source.
        force: Download again even if the file matches its record.

    Returns:
        dict: All records in the manifest, keyed by source name.
    """
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "SOURCES.json"
    records = load_sources(manifest)
    for name, source in sources.items():
        path = out / source["file"]
        record = records.get(name)
        # Skip a file that is on disk and still matches its recorded hash;
        # anything else (missing, changed or never recorded) is fetched.
        if (
            not force
            and record
            and path.exists()
            and sha256_of(path) == record["sha256"]
        ):
            print(f"{name}: {path.name} present, SHA-256 matches; skipped")
            continue
        print(f"{name}: downloading {source['url']}")
        fields = {
            k: v for k, v in source.items() if k not in ("url", "file", "headers")
        }
        records[name] = {
            **download(source["url"], path, source.get("headers")),
            **extra,
            **fields,
        }
        manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(
            f"{name}: {records[name]['bytes']:,} bytes, "
            f"sha256 {records[name]['sha256']}"
        )
    return records
