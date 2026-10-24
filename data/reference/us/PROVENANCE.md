# US reference layers: provenance

Layers the pipeline joins positions against. Everything here is
committed and rebuilt by `scripts/build_us_reference.py` from downloads
made by `scripts/fetch_us.py` (URL, retrieval time and SHA-256 recorded in
`data/raw/reference/SOURCES.json`, which is not committed).

## `land.geojson`

- **Source:** Natural Earth 10m physical land
  (`https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip`),
  downloaded 2026-10-24, SHA-256
  `e547d749445eaa0964aba76738090ec88f5e63c4585122170f98c67a7ea922dc`.
- **Licence:** public domain. Natural Earth asks for, but does not
  require, the credit "Made with Natural Earth".
- **Processing:** clipped to the study box (71.5W to 65W, 40N to 45N) plus
  0.5 degrees on every side, merged into one MultiPolygon, coordinates
  snapped to 1e-6 degrees. One feature, in longitude and latitude
  (EPSG:4326).
- **Limits (observed on the tuning block, 2026-10-24):** the coastline is
  too coarse for harbours. 35% of fishing-vessel positions fall inside
  this land, mostly moored boats, and it fills in upriver harbours
  (Portsmouth NH, the Merrimack) and the Cape Cod Canal. The pipeline
  therefore does not use it as a land mask; it is kept for distance
  offshore, where the same caveat applies near harbours.

## `closures.geojson`

The dated geofence table: one feature per rule, in longitude and
latitude, coordinates snapped to 1e-6 degrees. Read it with
`vessels.io.reference.read_closures`; the rule semantics are in
`vessels.core.geofence`.

- **Properties:** `area_id`, `name`, `kind` (`closure`, `exemption`,
  `suspension`), `parent` (the closure an exemption cuts out of, or the
  exemption a suspension switches off), `gear` (banned by a closure,
  allowed by an exemption), `valid_from` (included), `valid_to`
  (excluded; null while in force), `season` (inclusive month-day window
  or null), `timezone` (all `America/New_York`: regulation dates are
  local days), `citation`, `source_type` (`cfr`, `efp`, `asmfc`),
  `geometry_source` (`cfr`, `garfo_shapefile`, `fr_notice`, `parent`),
  `status` (`observed` or `provisional`) and `note`.
- **Rows:** every Habitat Management Area (50 CFR 648.370) and Dedicated
  Habitat Research Area (648.371) that bans gear inside the study box,
  all from Omnibus Habitat Amendment 2 (83 FR 15240, in force
  2018-04-09), with these exemptions and suspensions:
  - Western Gulf of Maine Shrimp Exemption Area (shrimp trawls), from
    2018-04-09, suspended 2019-2021 and 2022-2024 by the ASMFC northern
    shrimp moratorium (news releases of 16 November 2018 and
    17 December 2021);
  - Great South Channel hydraulic clam dredge exemption of 2018, which
    covered the HMA minus its north-east corner (the polygon GARFO's
    shapefile calls the "MBTG Closure") until 9 April 2019;
  - the McBlair, Old South (1 May-31 October) and Fishing Rip dredge
    exemption areas of the Clam Dredge Exemption Framework (85 FR 29870,
    in force 2020-06-18, no sunset);
  - the Coonamessett Farm Foundation exempted fishing permit for
    hydraulic clam dredging at Davis Bank East, in force 2024-08-01
    (89 FR 18376 for the area, 90 FR 26274 for the dates) until the
    revised permit of 2024-11-06.
- **Sources:** the eCFR point-in-time XML of 648.370 for 2018-04-09,
  2020-06-18 and 2024-07-01 and of 648.371 for 2024-07-01; the GARFO
  Habitat Management Area shapefile (2018-04-09); the two Federal
  Register notices; the two ASMFC news releases. URLs, retrieval times
  and SHA-256 are in `data/raw/us_closures/SOURCES.json` (not committed).
  The build fails if a source no longer says what a row relies on.
- **Licence:** the CFR, Federal Register and GARFO data are US
  Government works (17 U.S.C. 105). The ASMFC releases are cited for two
  dates, not redistributed.
- **Geometry:** read from the CFR point tables, except Eastern Maine and
  Closed Area II, whose edges follow the state-waters limit and the
  US-Canada boundary, which the CFR describes but does not list as points;
  those come from the shapefile (NAD83, treated as WGS84: the datums
  differ by about a metre here). The other HMAs match the shapefile
  (intersection over union 0.9998 or more).
- **Observed:** eCFR shows no change to 648.370(h) after 2020-06-18 (checked
  to 2026-10-01) and none to Stellwagen or Georges Bank DHRAs since 2018.
  The 2023 change to the Western Gulf of Maine HMA (point WGMH4 from
  70°15′ W to 70°00′ W, 88 FR 17397) corrected a typographical error, so
  the corrected point is used for every date.
- **Provisional:**
  - the moratorium rows: the ASMFC measure works through state landing
    rules, not the CFR, and its dates come from the news releases, not
    the addenda. 2018 and 2025 onwards are not yet sourced, so the shrimp
    exemption reads as open then;
  - the permit row: a permit, not a regulation; its area after 2024-11-06
    and the Rose and Crown area added in 2025 were not published as
    coordinates;
  - the gear assumption: AIS has no gear, so exemptions are applied to any
    towing vessel, and a vessel inside a closure may be using gear the
    closure allows.
- **Not modelled:** transit with gear stowed (allowed in every area; the
  detector's speed and persistence rules handle it) and the Restricted
  Roller Gear Area (a gear specification, not a closure).

## `SHA256SUMS`

Checksums of every layer. Check them with `sha256sum -c SHA256SUMS` from
this folder.
