"""Monthly cost of running the detectors on AWS, from measured work and a
dated list-price snapshot. Projected, not billed: nothing here was
deployed.

Two designs are priced at a chosen volume of positions a day:

- **per message**, the collar-telemetry pattern: each position is one
  SQS FIFO message (sent, received and deleted), one Lambda invocation
  and one transactional DynamoDB write of the vessel's state;
- **the service** as built (``cloud.service``): the feed lands as one
  object per window, and one tick per window runs the micro-batch stream
  and commits its checkpoint and alerts as objects. Priced on Lambda
  (stateless, so each tick loads the checkpoint first) and on one
  always-on Fargate task (state kept in memory).

Transport alternatives for the feed are priced on their own
(``transports``).

Work is measured on a laptop over the extract (``scripts/
measure_cloud.py``) and scaled to the target volume in proportion to
positions. That holds for work per position and per tick if the state the
stream carries grows with the traffic, which is an assumption: the
extract holds one group's fishing vessels only (provisional). CPU seconds
on the laptop stand for vCPU seconds on Arm (provisional), and Lambda
gives one vCPU at 1,769 MB, so CPU-bound work costs the same GB-seconds
at any smaller memory size. Prices are list rates before free tiers.
"""

import math
from dataclasses import dataclass

DAYS_PER_MONTH = 365.25 / 12
HOURS_PER_MONTH = DAYS_PER_MONTH * 24
SECONDS_PER_DAY = 86_400
# Lambda allocates one vCPU at 1,769 MB (AWS Lambda developer guide).
LAMBDA_VCPU_GB = 1769 / 1024
# SQS bills each 64 KB of a request's payload as one request.
SQS_CHUNK_BYTES = 64 * 1024
SQS_REQUESTS_PER_MESSAGE = 3
# Kinesis provisioned: a PUT payload unit is 25 KB; a shard takes 1 MB/s
# or 1,000 records/s.
KINESIS_UNIT_BYTES = 25 * 1024
KINESIS_SHARD_BYTES = 2**20
KINESIS_SHARD_RECORDS = 1000
# A per-message state write: one item of at most 1 KB, written in a
# transaction (2 WRU per KB) after a strongly consistent read (1 RRU per
# 4 KB). Item size assumed (provisional).
STATE_WRITE_UNITS = 2
STATE_READ_UNITS = 1
# One spool object per window: the producer's PUT, then the processor's
# LIST and GET. Deletes are free.
SPOOL_PUTS, SPOOL_LISTS, SPOOL_GETS = 1, 1, 1
FARGATE_GB = 2
LAMBDA_MAX_SECONDS = 900


@dataclass(frozen=True)
class Work:
    """Work measured on the extract, per position and per tick.

    Attributes:
        positions_per_day: Arrivals a day in the measured run.
        tick_minutes: The service's window.
        tick_cpu_per_position: Service CPU seconds per position, ticks
            and their checkpoint and sink I/O included.
        tick_seconds_p95: 95th-percentile wall time of a tick.
        puts_per_tick: Checkpoint and alert objects written per tick.
        load_seconds: Time to load and restore the committed checkpoint.
        load_files: Objects read to load it.
        checkpoint_bytes: Mean committed checkpoint size.
        single_row_seconds: Mean time to push one position alone.
        json_bytes: Mean size of one position as JSON.
        batch_cpu_per_position: CPU seconds per position, detectors once
            over the whole span.
        hourly_cpu_per_position: CPU seconds per position, stream fed by
            the hour.
    """

    positions_per_day: float
    tick_minutes: int
    tick_cpu_per_position: float
    tick_seconds_p95: float
    puts_per_tick: float
    load_seconds: float
    load_files: float
    checkpoint_bytes: float
    single_row_seconds: float
    json_bytes: float
    batch_cpu_per_position: float
    hourly_cpu_per_position: float

    @classmethod
    def from_measured(cls, measured: dict) -> "Work":
        """Read the measurement script's output.

        Args:
            measured: ``results/cloud/measured.json``.

        Returns:
            Work: The figures the model uses.
        """
        service = measured["service"]
        arrivals = measured["arrivals"]
        return cls(
            positions_per_day=arrivals / len(measured["days"]),
            tick_minutes=service["window_minutes"],
            tick_cpu_per_position=service["cpu_seconds"] / arrivals,
            tick_seconds_p95=service["tick_seconds"]["p95"],
            puts_per_tick=service["puts_per_tick"],
            load_seconds=service["load"]["seconds"]["mean"],
            load_files=service["load"]["files"],
            checkpoint_bytes=service["checkpoint_bytes"]["mean"],
            single_row_seconds=measured["single_row"]["seconds"]["mean"],
            json_bytes=measured["json_bytes_per_position"],
            batch_cpu_per_position=measured["batch"]["cpu_seconds"]
            / measured["positions"],
            hourly_cpu_per_position=measured["hourly"]["cpu_seconds"] / arrivals,
        )


@dataclass(frozen=True)
class Line:
    """One line of a monthly bill.

    Attributes:
        name: What is billed.
        quantity: Units a month.
        price: Snapshot name of the unit price.
        basis: How the quantity was found.
    """

    name: str
    quantity: float
    price: str
    basis: str


def unit_prices(snapshot: dict) -> dict[str, float]:
    """Unit prices by name from a price snapshot.

    Args:
        snapshot: As written by ``scripts/fetch_prices.py``.

    Returns:
        dict[str, float]: USD per unit.
    """
    return {name: entry["usd"] for name, entry in snapshot["prices"].items()}


def bill(lines: list[Line], prices: dict[str, float]) -> dict:
    """Price bill lines.

    Args:
        lines: The lines.
        prices: ``unit_prices``.

    Returns:
        dict: Per line its quantity, unit price, USD and basis; and the
        total USD.
    """
    priced = [
        {
            "name": line.name,
            "quantity": line.quantity,
            "unit_usd": prices[line.price],
            "usd": line.quantity * prices[line.price],
            "basis": line.basis,
        }
        for line in lines
    ]
    return {"lines": priced, "total_usd": sum(line["usd"] for line in priced)}


def scale(work: Work, positions_per_day: float) -> float:
    """How many times the measured traffic a volume is.

    Args:
        work: Measured work.
        positions_per_day: Target volume.

    Returns:
        float: The ratio.
    """
    return positions_per_day / work.positions_per_day


def ticks_per_month(work: Work) -> float:
    """Service ticks in a month, one per window.

    Args:
        work: Measured work.

    Returns:
        float: Ticks.
    """
    return DAYS_PER_MONTH * 24 * 60 / work.tick_minutes


def per_message(work: Work, positions_per_day: float) -> list[Line]:
    """The collar-telemetry pattern: one message, invocation and state
    transaction per position.

    Args:
        work: Measured work.
        positions_per_day: Target volume.

    Returns:
        list[Line]: The monthly lines.
    """
    positions = positions_per_day * DAYS_PER_MONTH
    chunks = math.ceil(work.json_bytes / SQS_CHUNK_BYTES)
    return [
        Line(
            "SQS FIFO requests",
            positions * SQS_REQUESTS_PER_MESSAGE * chunks,
            "sqs_fifo_request",
            "send, receive and delete per position",
        ),
        Line(
            "Lambda requests",
            positions,
            "lambda_arm_request",
            "one invocation per position",
        ),
        Line(
            "Lambda compute (GB-s)",
            positions * work.single_row_seconds * LAMBDA_VCPU_GB,
            "lambda_arm_gb_second",
            "measured single-row push, in memory (state load excluded)",
        ),
        Line(
            "DynamoDB write units",
            positions * STATE_WRITE_UNITS,
            "dynamodb_write_request_unit",
            "one transactional state write per position (1 KB item assumed)",
        ),
        Line(
            "DynamoDB read units",
            positions * STATE_READ_UNITS,
            "dynamodb_read_request_unit",
            "one consistent state read per position",
        ),
    ]


def _objects(work: Work, positions_per_day: float, load: bool) -> list[Line]:
    """Object-store lines the service pays on any compute.

    Args:
        work: Measured work.
        positions_per_day: Target volume.
        load: Whether each tick reads the checkpoint back (stateless
            compute).

    Returns:
        list[Line]: The monthly lines.
    """
    ticks = ticks_per_month(work)
    gets = SPOOL_GETS + (work.load_files if load else 0)
    return [
        Line(
            "S3 PUT and LIST requests",
            ticks * (SPOOL_PUTS + SPOOL_LISTS + work.puts_per_tick),
            "s3_put_request",
            "spool object and listing per tick; measured checkpoint and alert writes",
        ),
        Line(
            "S3 GET requests",
            ticks * gets,
            "s3_get_request",
            "spool object per tick" + ("; measured checkpoint files" if load else ""),
        ),
        Line(
            "S3 storage (GB-month)",
            work.checkpoint_bytes * scale(work, positions_per_day) / 2**30,
            "s3_storage_gb_month",
            "measured mean checkpoint, scaled with traffic",
        ),
    ]


def duty(work: Work, positions_per_day: float) -> float:
    """Busy share of one vCPU the service needs, ticks only.

    Above 1 one process cannot keep up and the stream would have to be
    split between processes, which is not built.

    Args:
        work: Measured work.
        positions_per_day: Target volume.

    Returns:
        float: vCPUs kept busy on average.
    """
    return positions_per_day * work.tick_cpu_per_position / SECONDS_PER_DAY


def tick_seconds(work: Work, positions_per_day: float, load: bool) -> float:
    """A slow (95th-percentile) tick's wall time at a volume.

    Args:
        work: Measured work.
        positions_per_day: Target volume.
        load: Whether the tick loads the checkpoint first.

    Returns:
        float: Seconds; a tick longer than its window falls behind.
    """
    factor = scale(work, positions_per_day)
    return factor * (work.tick_seconds_p95 + (work.load_seconds if load else 0))


def lambda_keeps_up(work: Work, positions_per_day: float) -> bool:
    """Whether a slow Lambda tick ends within its window and Lambda's
    time limit, and the function's memory holds the checkpoint (a lower
    bound on the state it loads).

    Args:
        work: Measured work.
        positions_per_day: Target volume.

    Returns:
        bool: True if the service on Lambda keeps up as built.
    """
    limit = min(work.tick_minutes * 60, LAMBDA_MAX_SECONDS)
    state = work.checkpoint_bytes * scale(work, positions_per_day)
    return (
        tick_seconds(work, positions_per_day, load=True) <= limit
        and state < LAMBDA_VCPU_GB * 2**30
    )


def fargate_keeps_up(work: Work, positions_per_day: float) -> bool:
    """Whether one task's vCPU keeps up and its memory holds the
    checkpoint (a lower bound on the state in memory).

    Args:
        work: Measured work.
        positions_per_day: Target volume.

    Returns:
        bool: True if the service on one Fargate task keeps up as built.
    """
    state = work.checkpoint_bytes * scale(work, positions_per_day)
    return duty(work, positions_per_day) < 1 and state < FARGATE_GB * 2**30


def service_lambda(work: Work, positions_per_day: float) -> list[Line]:
    """The service with each tick a Lambda invocation that loads the
    checkpoint, runs and commits.

    Args:
        work: Measured work.
        positions_per_day: Target volume.

    Returns:
        list[Line]: The monthly lines.
    """
    ticks = ticks_per_month(work)
    cpu = (
        positions_per_day * DAYS_PER_MONTH * work.tick_cpu_per_position
        + ticks * work.load_seconds * scale(work, positions_per_day)
    )
    return [
        Line("Lambda requests", ticks, "lambda_arm_request", "one invocation per tick"),
        Line(
            "Lambda compute (GB-s)",
            cpu * LAMBDA_VCPU_GB,
            "lambda_arm_gb_second",
            "measured tick CPU and checkpoint load, scaled with traffic",
        ),
    ] + _objects(work, positions_per_day, load=True)


def service_fargate(work: Work, positions_per_day: float) -> list[Line]:
    """The service as one always-on Fargate task holding state in memory.

    Args:
        work: Measured work.
        positions_per_day: Target volume.

    Returns:
        list[Line]: The monthly lines.
    """
    return [
        Line(
            "Fargate vCPU-hours",
            HOURS_PER_MONTH,
            "fargate_arm_vcpu_hour",
            "one 1 vCPU task, always on",
        ),
        Line(
            "Fargate GB-hours",
            HOURS_PER_MONTH * FARGATE_GB,
            "fargate_arm_gb_hour",
            f"{FARGATE_GB} GB, always on",
        ),
    ] + _objects(work, positions_per_day, load=False)


def transports(work: Work, positions_per_day: float) -> dict[str, list[Line]]:
    """Ways to carry the feed to the processor, each with its own lines.

    Args:
        work: Measured work.
        positions_per_day: Target volume.

    Returns:
        dict[str, list[Line]]: Lines by transport.
    """
    positions = positions_per_day * DAYS_PER_MONTH
    windows = ticks_per_month(work)
    per_message_ = math.floor(SQS_CHUNK_BYTES / work.json_bytes)
    per_record = math.floor(KINESIS_UNIT_BYTES / work.json_bytes)
    records_per_second = positions_per_day / per_record / SECONDS_PER_DAY
    bytes_per_second = positions_per_day * work.json_bytes / SECONDS_PER_DAY
    shards = max(
        1,
        math.ceil(records_per_second / KINESIS_SHARD_RECORDS),
        math.ceil(bytes_per_second / KINESIS_SHARD_BYTES),
    )
    return {
        "SQS FIFO, one position per message": [
            Line(
                "SQS FIFO requests",
                positions * SQS_REQUESTS_PER_MESSAGE,
                "sqs_fifo_request",
                "send, receive and delete per position",
            )
        ],
        "SQS standard, packed into 64 KB messages": [
            Line(
                "SQS standard requests",
                math.ceil(positions / per_message_) * SQS_REQUESTS_PER_MESSAGE,
                "sqs_standard_request",
                f"{per_message_} positions per message",
            )
        ],
        "Kinesis provisioned, packed into 25 KB records": [
            Line(
                "Kinesis shard-hours",
                shards * HOURS_PER_MONTH,
                "kinesis_shard_hour",
                f"{shards} shard(s) for the mean rate",
            ),
            Line(
                "Kinesis PUT payload units",
                math.ceil(positions / per_record),
                "kinesis_put_payload_unit",
                f"{per_record} positions per record",
            ),
        ],
        "S3, one object per window (as built)": [
            Line(
                "S3 PUT and LIST requests",
                windows * (SPOOL_PUTS + SPOOL_LISTS),
                "s3_put_request",
                f"one object a {work.tick_minutes} minutes",
            ),
            Line(
                "S3 GET requests",
                windows * SPOOL_GETS,
                "s3_get_request",
                "read once",
            ),
        ],
    }
