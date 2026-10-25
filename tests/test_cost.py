"""Tests for the cost model: hand-worked bills on round numbers, the
keep-up checks at their boundaries, and the committed cost note
rebuilding byte for byte from committed files."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cloud import cost

ROOT = Path(__file__).resolve().parents[1]

# Two days of 1,000 arrivals each; every figure chosen to work by hand.
MEASURED = {
    "days": ["2024-01-01", "2024-01-02"],
    "positions": 2000,
    "arrivals": 2000,
    "json_bytes_per_position": 200.0,
    "batch": {"seconds": 0.2, "cpu_seconds": 0.2},
    "hourly": {"seconds": 1.0, "cpu_seconds": 1.0},
    "service": {
        "window_minutes": 10,
        "cpu_seconds": 2.0,
        "tick_seconds": {"p95": 0.5},
        "puts_per_tick": 4.0,
        "load": {"seconds": {"mean": 0.1}, "files": 10.0},
        "checkpoint_bytes": {"mean": 2.0**20},
    },
    "single_row": {"seconds": {"mean": 0.01}},
}
PRICES = {
    "lambda_arm_request": 2e-7,
    "lambda_arm_gb_second": 1e-5,
    "sqs_standard_request": 4e-7,
    "sqs_fifo_request": 5e-7,
    "kinesis_shard_hour": 0.015,
    "kinesis_put_payload_unit": 1.4e-8,
    "dynamodb_write_request_unit": 6.25e-7,
    "dynamodb_read_request_unit": 1.25e-7,
    "fargate_arm_vcpu_hour": 0.03,
    "fargate_arm_gb_hour": 0.004,
    "s3_put_request": 5e-6,
    "s3_get_request": 4e-7,
    "s3_storage_gb_month": 0.023,
}
MONTH = 365.25 / 12
TICKS = MONTH * 144
LAMBDA_GB = 1769 / 1024


@pytest.fixture(name="work")
def fixture_work():
    """Work read from the hand-made measurement."""
    return cost.Work.from_measured(MEASURED)


def quantities(lines):
    """Quantities by line name.

    Args:
        lines: Bill lines.

    Returns:
        dict: Quantity by name.
    """
    return {line.name: line.quantity for line in lines}


def test_work_reads_rates_per_position(work):
    """Measured totals become per-position and per-day rates."""
    assert work.positions_per_day == 1000
    assert work.tick_cpu_per_position == pytest.approx(0.001)
    assert work.batch_cpu_per_position == pytest.approx(1e-4)
    assert work.hourly_cpu_per_position == pytest.approx(5e-4)
    assert cost.scale(work, 15_000) == 15


def test_per_message_bills_each_position(work):
    """The collar pattern pays three queue requests, one invocation, a
    transaction and a read per position."""
    got = quantities(cost.per_message(work, 1000))
    positions = 1000 * MONTH
    assert got["SQS FIFO requests"] == pytest.approx(3 * positions)
    assert got["Lambda requests"] == pytest.approx(positions)
    assert got["Lambda compute (GB-s)"] == pytest.approx(positions * 0.01 * LAMBDA_GB)
    assert got["DynamoDB write units"] == pytest.approx(2 * positions)
    assert got["DynamoDB read units"] == pytest.approx(positions)


def test_service_lambda_pays_ticks_and_checkpoint_loads(work):
    """A stateless tick pays its scaled CPU, a checkpoint load and the
    measured object reads and writes."""
    got = quantities(cost.service_lambda(work, 2000))
    cpu = 2000 * MONTH * 0.001 + TICKS * 0.1 * 2
    assert got["Lambda requests"] == pytest.approx(TICKS)
    assert got["Lambda compute (GB-s)"] == pytest.approx(cpu * LAMBDA_GB)
    assert got["S3 PUT and LIST requests"] == pytest.approx(TICKS * (1 + 1 + 4))
    assert got["S3 GET requests"] == pytest.approx(TICKS * (1 + 10))
    assert got["S3 storage (GB-month)"] == pytest.approx(2 / 1024)


def test_service_fargate_is_one_task_without_loads(work):
    """An always-on task pays a month of one vCPU and 2 GB, and reads only
    the spool."""
    got = quantities(cost.service_fargate(work, 1000))
    assert got["Fargate vCPU-hours"] == pytest.approx(MONTH * 24)
    assert got["Fargate GB-hours"] == pytest.approx(MONTH * 48)
    assert got["S3 GET requests"] == pytest.approx(TICKS)


def test_bill_prices_and_totals():
    """Each line is quantity times unit price, and the total their sum."""
    lines = [
        cost.Line("a", 1e6, "lambda_arm_request", ""),
        cost.Line("b", 2e6, "sqs_fifo_request", ""),
    ]
    priced = cost.bill(lines, PRICES)
    assert [line["usd"] for line in priced["lines"]] == pytest.approx([0.2, 1.0])
    assert priced["total_usd"] == pytest.approx(1.2)


def test_transports_pack_positions(work):
    """Packed messages and records hold as many whole positions as fit,
    and Kinesis adds a shard per started MiB a second."""
    got = cost.transports(work, 1e9)
    positions = 1e9 * MONTH
    sqs = quantities(got["SQS standard, packed into 64 KB messages"])
    assert sqs["SQS standard requests"] == 3 * -(-positions // 327)
    kinesis = quantities(got["Kinesis provisioned, packed into 25 KB records"])
    # 1e9 x 200 B a day is 2.21 MiB/s: three shards.
    assert kinesis["Kinesis shard-hours"] == pytest.approx(3 * MONTH * 24)
    assert kinesis["Kinesis PUT payload units"] == -(-positions // 128)
    low = quantities(
        cost.transports(work, 1000)["Kinesis provisioned, packed into 25 KB records"]
    )
    assert low["Kinesis shard-hours"] == pytest.approx(MONTH * 24)


def test_fargate_keeps_up_below_one_vcpu():
    """One task keeps up only while its ticks need less than one vCPU."""
    light = MEASURED | {
        "service": MEASURED["service"] | {"checkpoint_bytes": {"mean": 0.0}}
    }
    work = cost.Work.from_measured(light)
    full = 86_400 / 0.001
    assert cost.duty(work, full) == pytest.approx(1)
    assert cost.fargate_keeps_up(work, full * 0.99)
    assert not cost.fargate_keeps_up(work, full)


def test_fargate_needs_memory_for_the_checkpoint(work):
    """One task fails once the scaled checkpoint fills its 2 GB."""
    assert cost.fargate_keeps_up(work, 2047 * 1000)
    assert not cost.fargate_keeps_up(work, 2048 * 1000)


def test_lambda_keeps_up_while_a_slow_tick_fits_its_window(work):
    """A slow tick (0.6 s measured, load included) fits a 600 s window up
    to 1,000 times the traffic."""
    assert cost.tick_seconds(work, 1e6, load=True) == pytest.approx(600)
    assert cost.lambda_keeps_up(work, 1e6)
    assert not cost.lambda_keeps_up(work, 1.001e6)


def test_lambda_needs_memory_for_the_checkpoint():
    """A function fails once the scaled checkpoint fills its 1,769 MB,
    even with time to spare."""
    fast = MEASURED | {"service": MEASURED["service"] | {"tick_seconds": {"p95": 0.0}}}
    fast["service"]["load"] = {"seconds": {"mean": 0.0}, "files": 10.0}
    work = cost.Work.from_measured(fast)
    assert cost.lambda_keeps_up(work, 1768 * 1000)
    assert not cost.lambda_keeps_up(work, 1769 * 1000)


def test_committed_note_rebuilds_byte_for_byte(tmp_path):
    """make cost from the committed inputs gives the committed note."""
    env = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        [sys.executable, "scripts/report_cost.py", "--out", str(tmp_path)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    for name in ("COST.md", "cost.json"):
        committed = (ROOT / "results" / "cloud" / name).read_bytes()
        assert (tmp_path / name).read_bytes() == committed, name
    snapshot = json.loads((ROOT / "data" / "prices" / "aws-us-east-1.json").read_text())
    assert set(snapshot["prices"]) == set(PRICES)
