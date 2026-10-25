"""Fetch a dated snapshot of the AWS list prices the cost note uses.

Optional: the snapshot is committed (``data/prices/aws-us-east-1.json``),
so ``make cost`` never needs the network. Reads AWS's public Price List
files (no account or credentials) for each service the cost note prices,
keeps only the usage types it needs, and records each price with its
offer file and publication date.

Each price is the lowest tier with a non-zero rate: what a small workload
pays once any allowance written into the price list is used up. The cost
note prices list rates without free tiers; allowances the price list
holds are recorded under ``free_tier`` for the text.

Adapted from collar-telemetry @ v0.1.0 (scripts/fetch_prices.py): region
us-east-1, whose usage types carry no region prefix except Fargate's, and
the services of an AIS-scale design.
"""

import argparse
import json
from datetime import date
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "prices" / "aws-us-east-1.json"
HOST = "https://pricing.us-east-1.amazonaws.com"
REGION = "us-east-1"

# Name in the snapshot: (Price List service code, usage type).
WANTED = {
    "lambda_arm_request": ("AWSLambda", "Request-ARM"),
    "lambda_arm_gb_second": ("AWSLambda", "Lambda-GB-Second-ARM"),
    "sqs_standard_request": ("AWSQueueService", "Requests-RBP"),
    "sqs_fifo_request": ("AWSQueueService", "Requests-FIFO-RBP"),
    "kinesis_shard_hour": ("AmazonKinesis", "Storage-ShardHour"),
    "kinesis_put_payload_unit": ("AmazonKinesis", "PutRequestPayloadUnits"),
    "dynamodb_write_request_unit": ("AmazonDynamoDB", "WriteRequestUnits"),
    "dynamodb_read_request_unit": ("AmazonDynamoDB", "ReadRequestUnits"),
    "fargate_arm_vcpu_hour": ("AmazonECS", "USE1-Fargate-ARM-vCPU-Hours:perCPU"),
    "fargate_arm_gb_hour": ("AmazonECS", "USE1-Fargate-ARM-GB-Hours"),
    "s3_put_request": ("AmazonS3", "Requests-Tier1"),
    "s3_get_request": ("AmazonS3", "Requests-Tier2"),
    "s3_storage_gb_month": ("AmazonS3", "TimedStorage-ByteHrs"),
}


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        argparse.Namespace: ``out``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def get_json(path: str) -> dict:
    """Fetch one Price List file.

    Args:
        path: Path on the Price List host.

    Returns:
        dict: The parsed file.
    """
    with urlopen(HOST + path, timeout=300) as response:
        return json.load(response)


def pick(offer: dict, usage_type: str) -> tuple[dict, int | None]:
    """The lowest non-zero tier of a usage type, and any free allowance
    written into the price list as a zero-priced tier.

    Args:
        offer: One service's regional offer file.
        usage_type: The usage type, such as ``Request-ARM``.

    Returns:
        tuple[dict, int | None]: The price entry, and the free quantity
        per month, or None if the price list has no free tier for it.

    Raises:
        LookupError: If the usage type is missing or has no on-demand
            price.
    """
    skus = [
        sku
        for sku, product in offer["products"].items()
        if product["attributes"].get("usagetype") == usage_type
    ]
    if len(skus) != 1:
        raise LookupError(f"{usage_type}: {len(skus)} products")
    dimensions = [
        dimension
        for term in offer["terms"]["OnDemand"][skus[0]].values()
        for dimension in term["priceDimensions"].values()
    ]
    dimensions.sort(key=lambda d: float(d["beginRange"]))
    free = [d for d in dimensions if float(d["pricePerUnit"]["USD"]) == 0]
    paid = [d for d in dimensions if float(d["pricePerUnit"]["USD"]) > 0]
    if not paid:
        raise LookupError(f"{usage_type}: no paid tier")
    entry = {
        "sku": skus[0],
        "usage_type": usage_type,
        "unit": paid[0]["unit"],
        "usd": float(paid[0]["pricePerUnit"]["USD"]),
        "description": paid[0]["description"],
    }
    return entry, int(float(free[0]["endRange"])) if free else None


def main():
    """Fetch each service's us-east-1 offer file and write the snapshot."""
    args = parse_args()
    services = get_json("/offers/v1.0/aws/index.json")["offers"]
    offers = {}
    for code in sorted({code for code, _ in WANTED.values()}):
        regions = get_json(services[code]["currentRegionIndexUrl"])["regions"]
        path = regions[REGION]["currentVersionUrl"]
        offers[code] = (path, get_json(path))
    prices, free_tier = {}, {}
    for name, (code, usage_type) in WANTED.items():
        path, offer = offers[code]
        entry, free = pick(offer, usage_type)
        prices[name] = {
            "service": code,
            **entry,
            "offer": HOST + path,
            "published": offer["publicationDate"],
        }
        if free is not None:
            free_tier[name] = {"free_per_month": free, "source": HOST + path}
    snapshot = {
        "region": REGION,
        "currency": "USD",
        "retrieved": date.today().isoformat(),
        "prices": prices,
        "free_tier": dict(sorted(free_tier.items())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    for name, entry in prices.items():
        print(f"{name}: {entry['usd']} USD per {entry['unit']}")


if __name__ == "__main__":
    main()
