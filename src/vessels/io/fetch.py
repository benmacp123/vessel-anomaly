"""Hashed downloads with a provenance manifest, shared by the fetch scripts.

Each source is streamed to disk, hashed on the way, checked against the
server's length and recorded in a ``SOURCES.json`` beside the files, so a
rerun can skip anything already present and unchanged.
"""

import hashlib
import http.client
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = "vessel-anomaly portfolio project (open data fetch)"
CHUNK = 1 << 20
# Large files over a slow link stall now and then; each retry resumes.
RETRIES = 8


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


def _fetch_into(request: urllib.request.Request, partial: Path, timeout: float):
    """Append one response to a partial file, resuming where it stops.

    Args:
        request: The request; a ``Range`` header is added when the partial
            file already holds bytes.
        partial: The ``.part`` file.
        timeout: Seconds to wait for each read.

    Returns:
        tuple: The expected total size (None if not announced) and the
        server's ``Last-Modified`` header.

    Raises:
        ConnectionError: If the body ends before the announced length.
    """
    offset = partial.stat().st_size if partial.exists() else 0
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        # A server that ignores the range sends the whole file again.
        if offset and response.status != 206:
            offset = 0
        length = response.headers.get("Content-Length")
        modified = response.headers.get("Last-Modified")
        with partial.open("ab" if offset else "wb") as handle:
            while chunk := response.read(CHUNK):
                handle.write(chunk)
    expected = None if length is None else offset + int(length)
    # http.client ends a dropped body quietly instead of raising, so a
    # short file is treated as a dropped connection and resumed.
    if expected is not None and partial.stat().st_size < expected:
        raise ConnectionError(
            f"body ended at {partial.stat().st_size:,} of {expected:,} bytes"
        )
    return expected, modified


def download(
    url: str,
    path: Path,
    headers: dict | None = None,
    retries: int = RETRIES,
    timeout: float = 120,
) -> dict:
    """Stream a URL to a file, resuming after stalls, and hash it.

    The data go to a ``.part`` file first and are renamed only once the
    byte count matches the server's ``Content-Length``, so an interrupted
    run never leaves a truncated file under the final name. A stalled or
    dropped connection is retried with a ``Range`` request from where it
    stopped. The bytes are stored as sent: a gzip-encoded response stays
    gzip on disk.

    Args:
        url: Source URL.
        path: Destination file.
        headers: Extra request headers.
        retries: Further attempts after a timeout or dropped connection.
        timeout: Seconds to wait for each read.

    Returns:
        dict: The provenance record: URL, file name, retrieval time (UTC),
        size, SHA-256 and the server's ``Last-Modified`` header.

    Raises:
        OSError: If the download is shorter or longer than announced, or
            still failing after the retries.
        urllib.error.URLError: If the request fails.
    """
    partial = path.with_name(path.name + ".part")
    # A part file left by an earlier run may belong to an older version of
    # the source, so resuming happens only within this call.
    partial.unlink(missing_ok=True)
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, **(headers or {})}
        )
        try:
            expected, modified = _fetch_into(request, partial, timeout)
            break
        except (TimeoutError, ConnectionError, http.client.IncompleteRead) as error:
            if attempt == retries:
                raise OSError(f"{url}: failed after {retries} retries") from error
            print(f"{path.name}: {error!r}; resuming (retry {attempt + 1})")
            time.sleep(min(60, 5 * 2**attempt))
    size = partial.stat().st_size
    if expected is not None and size != expected:
        partial.unlink()
        raise OSError(f"{url}: got {size:,} bytes, expected {expected:,}")
    partial.replace(path)
    return {
        "url": url,
        "file": path.name,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": size,
        "sha256": sha256_of(path),
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
