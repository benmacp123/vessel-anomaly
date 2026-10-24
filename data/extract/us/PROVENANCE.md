# US extract: provenance and terms

Fishing-vessel AIS positions in the Gulf of Maine and on Georges Bank,
May and September 2024, cut from NOAA MarineCadastre and pseudonymised.
Everything the analysis needs is in this folder; no account, key or
download is required to reproduce the results.

## Source

- **Dataset:** NOAA Office for Coastal Management. (2024). AIS Vessel
  Traffic, daily point data (GeoParquet). Marine Cadastre.
  https://marinecadastre.gov. Files were fetched from
  `https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/`.
- **Origin:** US Coast Guard Nationwide AIS (NAIS), land-based receivers
  only. The NOAA FAQ states there is no satellite AIS and that coverage
  is unavailable "more than 40 to 50 miles from the coast". Positions are
  down-sampled to the nearest whole minute.
- **Per-file provenance:** the fetch script records each raw file's URL,
  retrieval time, size and SHA-256. The raw files are not committed.

## Terms

- **US Government material.** This extract consists predominantly of
  US Government material, which is not subject to copyright protection
  within the United States (17 U.S.C. § 105). This notice is given as
  17 U.S.C. § 403 requires. Foreign copyrights may apply.
- **Redistribution.** The NOAA AIS FAQ (May 2026) answers "Can I use and
  redistribute these data on other websites?" with "Yes, you may release
  derived products built from these data. Please cite your data source".
  This extract is such a derived product: a subset of rows and columns,
  with identities replaced.
- **USCG condition.** The same FAQ quotes the Coast Guard's data-sharing
  conditions, including that entities "shall not retransmit or
  redistribute AIS information (real-time or stored) in any form other
  than those intended for the disclosure as approved, and shall not
  charge a fee for its usage". The extract is free, historical and
  derived, and it carries no MMSI, name, IMO number or call sign.
- **Planning purposes only.** NOAA states the data "are intended for
  coastal and ocean planning purposes only" and not for regulatory or
  enforcement use. Nothing here identifies a vessel or alleges a breach of
  any rule; alerts are statistical flags for a portfolio study.

## What the extract holds

- **Area:** longitude −71.5 to −65, latitude 40 to 45.
- **Blocks:** tuning, 6–19 May and 2–15 September 2024; held out,
  20 May–2 June and 16–29 September 2024. The blocks were fixed before any
  of their days was downloaded.
- **Vessels:** AIS vessel types 30, 1001 and 1002 (fishing). Each block
  holds one half of the fleet, so no vessel appears in both blocks.
- **Identity:** `vessel_id` is the first 16 hex digits of
  HMAC-SHA256(salt, MMSI). The salt is secret, so the IDs cannot be
  reversed. A vessel's half follows from its ID: group A (tuning) when
  the first hex digit is even, B (held out) when it is odd.
- **`<block>/<day>.parquet`:** one file per day, zstd, sorted by every
  column starting with `vessel_id` and `timestamp`. Columns are
  `vessel_id`, `timestamp` (UTC), `lon`, `lat`, `sog`, `cog`, `heading`,
  `status`, `vessel_type`, `length` and `transceiver`. One rule is applied
  before hashing, because it needs the real MMSI: rows are dropped when the
  MMSI is a placeholder (123456789, 1193046), is not nine digits, or has a
  country code (MID) outside the ship range 201–775 (USCG NAVCEN). In the
  2024 block days every drop was for the MID; the prefixes seen nationally
  (8xx, 100, 191, 784) are mostly misconfigured or non-ship units, but a
  few move at ship speeds, so some real vessels with mistyped MMSIs are
  excluded. Otherwise rows are kept as published: other cleaning and
  de-duplication are part of the analysis.
- **`<block>/reception/<day>.parquet`:** counts from all traffic in the
  area, every vessel type and both halves of the fleet. There is one row
  per H3 resolution-6 cell and AIS class, giving `positions`, distinct
  `vessels`, `moving_intervals` and `long_intervals`. An interval runs
  from one position to the same vessel's next position that day and
  belongs to the cell and class where it starts; it counts as moving if
  it starts at 2 knots or more, and as long if it also exceeds 10
  minutes. Intervals across midnight are not counted. The coverage mask
  is built from these counts.
- **`counts.csv`:** rows and vessels per daily file, grid cells and
  all-traffic rows, and the block's fishing rows dropped per MMSI rule
  (`mmsi_placeholder_rows`, `mmsi_length_rows`, `mmsi_mid_rows`).
- **`SHA256SUMS`:** checksums of every Parquet file. Check them with
  `sha256sum -c SHA256SUMS` from this folder.

## Reproducing the extract

Reviewers do not need to: every result is computed from the committed
files. The owner can rebuild them with `scripts/fetch_us.py --blocks`
and then `scripts/build_us_extract.py`. A rebuild is byte for byte
identical only with the original salt and the pinned library versions.
Without the salt, the same steps give the same rows under different IDs
and a different vessel split.
