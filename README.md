# Vessel-behaviour anomaly detection

[![Python Applications test with GitHub Actions](https://github.com/benmacp123/vessel-anomaly/actions/workflows/main.yml/badge.svg)](https://github.com/benmacp123/vessel-anomaly/actions/workflows/main.yml)

Can a small, reproducible pipeline flag vessel behaviour worth a fisheries
analyst's attention, and how often is it right? The detectors cover fishing
inside a closure that bans the gear in use, unexplained AIS gaps, encounters
at sea and loitering, all run on public AIS tracks.

**Status:** early development. Nothing here is a result yet.

## Approach

- Every detector runs per track segment, not per MMSI, because MMSIs are
  shared and sometimes invalid.
- Closures are dated and gear-specific. Presence in a protected area is not
  treated as a signal.
- A coverage mask records where the receivers cannot see, and gaps are only
  reported inside it.
- Thresholds are tuned on one block of data, frozen, and scored once on a
  held-out block.

## How it works

The diagram shows the architecture as it stands and is updated as parts
are built. Solid boxes and arrows exist; dashed ones are planned. The
study region is the US Northeast (Gulf of Maine and Georges Bank) in 2024;
Norwegian data, with logbooks kept independently of AIS, is a
qualitative cross-check.

```mermaid
flowchart TB
    subgraph sources ["Public sources"]
        hais[("Norwegian AIS<br>Kystverket HAIS")]
        noaa[("US AIS<br>NOAA MarineCadastre")]
        fdir[("Fiskeridirektoratet<br>ERS logbooks, VMS")]
        refs[("Closures, coastline,<br>ports")]
    end

    fetch["Fetch scripts<br>scripts/fetch_us.py,<br>scripts/fetch_fiskeridir.py"]
    build["Extract builder<br>scripts/build_us_extract.py"]
    extract[("Committed US extract<br>data/extract/us/<br>tuning and held-out blocks,<br>reception grid")]
    io["Readers, fetch helpers<br>and extract schema<br>io/"]
    refbuild["Reference builder<br>scripts/build_us_reference.py"]
    refdata[("Committed reference layers<br>data/reference/us/<br>land, dated closures,<br>anchorages")]
    report["Cleaning report<br>scripts/report_cleaning.py<br>results/cleaning/"]
    covbuild["Coverage builder<br>scripts/build_us_coverage.py<br>results/coverage/"]
    closrep["Closure report<br>scripts/report_closures.py<br>results/closures/"]
    distbuild["Anchorage builder<br>scripts/build_us_distance.py<br>results/distance/"]

    subgraph core ["Detection core, no I/O (core/)"]
        clean["Clean and segment<br>cleaning.py, segment.py<br>GFW-style multi-track segments"]
        coverage["Coverage grid<br>coverage.py<br>covered cells per AIS class,<br>distance to edge"]
        geofence["Dated geofences<br>geofence.py<br>closures, exemptions and<br>suspensions by local day"]
        layers["Distance to shore and port<br>distance.py<br>signed shore distance,<br>anchorages from stationary vessels"]
        identity["Vessel identity<br>VMS call sign to AIS track"]
        vessel["Per-vessel detectors<br>gap inside coverage,<br>closure fishing, loitering"]
        encounter["Encounter detector<br>vessel pairs by H3 cell"]
        lifecycle["Episode lifecycle<br>adapted from collar-telemetry"]
    end

    sim["Arrival simulation and<br>synthetic injection<br>sim/"]
    stream["Micro-batch wrapper<br>stream/"]
    evaluation["Evaluation<br>frozen thresholds, held-out scoring,<br>logbook truth, alert rates<br>evaluation/"]
    cost["Cost model<br>cloud/"]

    fdir --> fetch
    noaa --> fetch
    refs --> fetch
    fetch --> build
    build --> extract
    io --> build
    fetch --> refbuild
    refbuild --> refdata
    hais -.-> io
    io --> distbuild
    layers --> distbuild
    distbuild --> refdata
    refdata --> io
    io --> closrep
    geofence --> closrep
    covbuild --> closrep
    extract --> io
    io --> report
    clean --> report
    io --> covbuild
    coverage --> covbuild
    coverage -.-> vessel
    io -.-> identity
    clean -.-> vessel
    clean -.-> encounter
    geofence -.-> vessel
    layers -.-> vessel
    layers -.-> encounter
    vessel -.-> lifecycle
    encounter -.-> lifecycle
    identity -.-> evaluation
    sim -.-> stream
    stream -.-> clean
    io -.-> stream
    lifecycle -.-> evaluation
    stream -.-> cost

    classDef planned stroke-dasharray: 5 5
    class identity,vessel,encounter,lifecycle,sim,stream,evaluation,cost planned
```

- **Clean and segment:** invalid MMSIs are dropped when the extract is
  built (the extract keeps only hashed IDs); cleaning then drops invalid
  positions and repeated messages and nulls invalid speeds and courses,
  as Global Fishing Watch does. The segmenter follows GFW's: a vessel can
  have several open tracks, each position joins the track whose
  dead-reckoned position it fits best, and a track closes after 24 hours
  of silence, so shared MMSIs and stray positions become separate
  segments. Every detector works on these segments. There is no land
  mask: the open coastline is too coarse for harbours (see
  [results/cleaning/](results/cleaning/)).
- **Coverage grid:** a cell counts as covered for an AIS class when
  moving vessels there are heard without dropouts: few of their
  position-to-position intervals exceed 10 minutes, with enough
  intervals to show it (95% Wilson upper bound of that share at most
  0.1). Cells with too little traffic to tell, and cells cut by the
  study box, are not covered. It is built from the tuning block's
  all-traffic counts only. Reception turned out to reach far offshore
  along shipping lanes, so the uncovered areas are mostly quiet water,
  plus a Class B edge on Georges Bank (see
  [results/coverage/](results/coverage/)).
- **Dated geofences:** each closure row bans named gear between two
  dates; exemption rows allow some gear back inside a closure (all year,
  for a season, or under a dated fishing permit), and suspension rows
  switch an exemption off, as the northern shrimp moratorium does for the
  Western Gulf of Maine shrimp area. Dates are local days in each row's
  time zone. AIS has no gear, so a towing vessel is assumed to use mobile
  bottom-tending gear, and towing inside an open exemption is not
  flagged. Sources, dates and limits are in
  [data/reference/us/PROVENANCE.md](data/reference/us/PROVENANCE.md);
  tuning-block counts are in [results/closures/](results/closures/).
- **Distance to shore and port:** distance to shore is signed (negative
  on land) and measured on the great circle to Natural Earth's 10m
  coastline, which is coarse in harbours but agrees with NASA's
  distance-to-coast grid (the one behind GFW's offshore rules) on more
  than 99% of positions at the 20 nm and 50 nm cuts. Ports are
  anchorages found as GFW finds them, from vessels that stay within
  0.5 km for 12 hours; four weeks of data cannot meet GFW's 20 vessels,
  so a cell needs 3 and must lie within 5 km of shore, so an offshore
  meeting spot is never taken for a port. Built from the tuning block
  only (see [results/distance/](results/distance/)).
- **Detectors:** a gap counts only inside measured receiver coverage;
  closure fishing needs gear-like speeds inside a closure that bans that
  gear on that date; encounters and loitering follow Global Fishing
  Watch's published rules and are compared with them, not claimed as new.
- **Vessel identity:** Norwegian logbooks and VMS carry call signs, not
  MMSIs, so VMS positions are matched to AIS tracks by time and place. A
  test week matched 84% of call signs to a unique MMSI, but none of that
  week's towed hauls fell inside the area covered by the AIS data, so
  Norway is a cross-check rather than the scored region.
- **Committed extract:** fishing-vessel positions for four tuning and
  four held-out weeks, with MMSIs replaced by salted hashes and each block
  holding a different half of the fleet. Provenance, terms and checksums
  are in [data/extract/us/PROVENANCE.md](data/extract/us/PROVENANCE.md).
- **Cost model:** estimates messages, state writes and compute for
  per-position against batched processing, rather than a deployed stack.

## Data

The committed US extract in [data/extract/us/](data/extract/us/) holds
fishing-vessel AIS positions (types 30, 1001 and 1002) for the Gulf of
Maine and Georges Bank, from NOAA MarineCadastre 2024. MMSIs are
replaced by salted hashes after dropping MMSIs whose country code (MID)
is outside the ship range 201–775, and each block holds a different half
of the fleet:

| Block | Days | Rows | Vessels |
| --- | --- | --- | --- |
| Tuning | 6–19 May, 2–15 September 2024 | 940,471 | 585 |
| Held out | 20 May–2 June, 16–29 September 2024 | 926,123 | 542 |

No vessel is in both blocks. Each block also carries a daily reception
grid of all-traffic counts per H3 cell and AIS class (positions,
vessels, and moving intervals with how many exceed 10 minutes), for the
coverage mask. The whole
extract is 25 MB, with no file over 1 MB. Sources, terms, column
definitions and checksums are in
[PROVENANCE.md](data/extract/us/PROVENANCE.md); counts per day are in
`counts.csv`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) 0.13.5 or later.

```sh
make install
source venv/bin/activate
make test lint format-check
```

`make help` lists every target.

## Licence

Code: MIT. Each dataset carries its own source and licence, recorded
alongside it.
