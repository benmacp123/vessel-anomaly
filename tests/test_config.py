"""Tests for the frozen configuration: it matches the code, and the check
catches any change."""

import json

import pytest

from vessels.evaluation import config


def test_committed_config_matches_code():
    """The committed configuration equals the code's defaults and inputs."""
    assert config.CONFIG.read_text() == config.dumps(config.frozen())


def test_recorded_hash_matches_committed_config():
    """The recorded hash is the committed configuration's own hash."""
    assert config.RECORDED.read_text().split()[0] == config.sha256_file(config.CONFIG)


def test_check_passes_on_fresh_freeze(tmp_path):
    """A freshly written configuration and hash pass the check."""
    path, recorded = tmp_path / "frozen.json", tmp_path / "frozen.sha256"
    value = config.write(path, recorded)
    assert config.check(path, recorded) == value
    assert value == config.sha256_file(path)


def test_check_rejects_changed_threshold(tmp_path):
    """A threshold edited in the file, not the code, fails the check."""
    path, recorded = tmp_path / "frozen.json", tmp_path / "frozen.sha256"
    config.write(path, recorded)
    data = json.loads(path.read_text())
    data["gaps"]["min_hours"] = 5.0
    path.write_text(config.dumps(data))
    with pytest.raises(RuntimeError, match="code defaults"):
        config.check(path, recorded)


def test_check_rejects_changed_code(tmp_path, monkeypatch):
    """A constant changed in code after the freeze fails the check."""
    path, recorded = tmp_path / "frozen.json", tmp_path / "frozen.sha256"
    config.write(path, recorded)
    monkeypatch.setattr(config, "MIN_EVENTS", 4)
    with pytest.raises(RuntimeError, match="code defaults"):
        config.check(path, recorded)


def test_check_rejects_wrong_recorded_hash(tmp_path):
    """A recorded hash from another configuration fails the check."""
    path, recorded = tmp_path / "frozen.json", tmp_path / "frozen.sha256"
    config.write(path, recorded)
    recorded.write_text("0" * 64 + "  frozen.json\n")
    with pytest.raises(RuntimeError, match="hash differs"):
        config.check(path, recorded)


def test_check_rejects_missing_freeze(tmp_path):
    """No freeze yet means no scoring."""
    with pytest.raises(RuntimeError, match="make freeze"):
        config.check(tmp_path / "frozen.json", tmp_path / "frozen.sha256")
