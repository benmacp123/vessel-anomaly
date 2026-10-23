"""Tests for the shared fetch helpers, run against local files through
``file://`` URLs so no network is needed."""

import hashlib
import json

from vessels.io.fetch import download, fetch_sources, load_sources, sha256_of


def test_download_records_hash_and_size(tmp_path):
    """A download's record carries the true SHA-256 and byte count."""
    payload = b"MeldingID;Tidspunkt(UTC)\n" * 100_000
    source = tmp_path / "source.zip"
    source.write_bytes(payload)
    target = tmp_path / "out.zip"

    record = download(source.as_uri(), target)

    assert target.read_bytes() == payload
    assert record["bytes"] == len(payload)
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["sha256"] == sha256_of(target)
    assert not target.with_name("out.zip.part").exists()


def test_load_sources_missing_file_is_empty(tmp_path):
    """No manifest yet means no records, so every source is fetched."""
    assert not load_sources(tmp_path / "SOURCES.json")


def test_fetch_sources_records_and_skips(tmp_path, capsys):
    """Records merge shared and per-source fields; a verified file is skipped."""
    source = tmp_path / "a.bin"
    source.write_bytes(b"abc")
    sources = {"a": {"url": source.as_uri(), "file": "a.bin", "licence": "X"}}
    out = tmp_path / "out"

    fetch_sources(sources, out, {"licence": "shared", "attribution": "Y"})
    manifest = json.loads((out / "SOURCES.json").read_text(encoding="utf-8"))
    assert (manifest["a"]["licence"], manifest["a"]["attribution"]) == ("X", "Y")

    fetch_sources(sources, out, {})
    assert "skipped" in capsys.readouterr().out
