"""Tests for the shared fetch helpers, run against local files through
``file://`` URLs and a local HTTP server, so no network is needed."""

import hashlib
import http.server
import json
import socket
import threading

from vessels.io import fetch
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


class FlakyRangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves ``PAYLOAD``, cutting the first response short and honouring
    ``Range`` afterwards, like a blob store over a poor link."""

    calls = []

    def do_GET(self):  # pylint: disable=invalid-name
        """Send the payload, or the requested tail of it."""
        start = 0
        if self.headers.get("Range"):
            start = int(self.headers["Range"].split("=")[1].rstrip("-"))
        body = PAYLOAD[start:]
        FlakyRangeHandler.calls.append(start)
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if len(FlakyRangeHandler.calls) == 1:
            # Promise the whole body, send a third of it, then hang up.
            self.wfile.write(body[: len(body) // 3])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        self.wfile.write(body)

    def log_message(self, *args):  # pylint: disable=arguments-differ
        """Keep test output quiet."""


PAYLOAD = bytes(range(256)) * 20_000


def test_download_resumes_after_dropped_connection(tmp_path, monkeypatch):
    """A connection cut mid-file is resumed with a Range request, not restarted."""
    monkeypatch.setattr(fetch.time, "sleep", lambda _: None)
    FlakyRangeHandler.calls = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FlakyRangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/file"
        record = download(url, tmp_path / "f.bin", timeout=5)
    finally:
        server.shutdown()
    assert (tmp_path / "f.bin").read_bytes() == PAYLOAD
    assert record["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    assert FlakyRangeHandler.calls[0] == 0 and FlakyRangeHandler.calls[1] > 0
