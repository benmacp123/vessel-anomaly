"""Tests for the Fiskeridirektoratet fetch script, run against local files
through ``file://`` URLs so no network is needed."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_fiskeridir.py"


@pytest.fixture(name="fetch")
def fixture_fetch():
    """Load the script as a module, since ``scripts/`` is not a package.

    Returns:
        module: The loaded fetch script.
    """
    spec = importlib.util.spec_from_file_location("fetch_fiskeridir", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_records_hash_and_size(fetch, tmp_path):
    """A download's record carries the true SHA-256 and byte count."""
    payload = b"MeldingID;Tidspunkt(UTC)\n" * 100_000
    source = tmp_path / "source.zip"
    source.write_bytes(payload)
    target = tmp_path / "out.zip"

    record = fetch.download(source.as_uri(), target)

    assert target.read_bytes() == payload
    assert record["bytes"] == len(payload)
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["sha256"] == fetch.sha256_of(target)
    assert not target.with_name("out.zip.part").exists()


def test_load_sources_missing_file_is_empty(fetch, tmp_path):
    """No manifest yet means no records, so every source is fetched."""
    assert not fetch.load_sources(tmp_path / "SOURCES.json")
