"""The US tuning and held-out blocks and the vessel split between them.

Fixed before any block day was downloaded: May and September 2024, two
weeks of each per block, with the earlier weeks of each month for tuning.
Vessels are split by their pseudonymous ID, so a reviewer can check from
the committed extracts alone that the blocks share no vessel.
"""

import hashlib
import hmac
from datetime import date, timedelta


def _days(first: date, last: date) -> list[date]:
    """Every day from ``first`` to ``last`` inclusive.

    Args:
        first: First day.
        last: Last day.

    Returns:
        list[date]: The days in order.
    """
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


BLOCKS = {
    "tuning": _days(date(2024, 5, 6), date(2024, 5, 19))
    + _days(date(2024, 9, 2), date(2024, 9, 15)),
    "heldout": _days(date(2024, 5, 20), date(2024, 6, 2))
    + _days(date(2024, 9, 16), date(2024, 9, 29)),
}
# The vessel group each block keeps.
BLOCK_GROUP = {"tuning": "A", "heldout": "B"}
VESSEL_ID_HEX = 16


def vessel_id(mmsi: int, salt: bytes) -> str:
    """The pseudonymous ID that replaces an MMSI in committed extracts.

    Args:
        mmsi: The MMSI.
        salt: The secret key, kept out of git.

    Returns:
        str: The first 16 hex digits of HMAC-SHA256(salt, MMSI).
    """
    digest = hmac.new(salt, str(mmsi).encode(), hashlib.sha256).hexdigest()
    return digest[:VESSEL_ID_HEX]


def vessel_group(vessel: str) -> str:
    """The block group of a pseudonymous vessel ID.

    Args:
        vessel: A ``vessel_id`` value.

    Returns:
        str: ``A`` (tuning) if its first hex digit is even, else ``B``.
    """
    return "A" if int(vessel[0], 16) % 2 == 0 else "B"
