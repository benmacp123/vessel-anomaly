"""Golden-file replays: each hand-built scenario still gives the events and
incidents recorded by ``make golden``."""

import json

import pytest

from golden import FOLDER, GOLDEN, build


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden(name):
    """A scenario's batch events and stream incidents match its file."""
    expected = json.loads((FOLDER / f"{name}.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(build(name))) == expected


def test_every_scenario_has_a_file():
    """No golden file is missing or left over."""
    assert {p.stem for p in FOLDER.glob("*.json")} == set(GOLDEN)
