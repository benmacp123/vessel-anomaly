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

## `SHA256SUMS`

Checksums of every layer. Check them with `sha256sum -c SHA256SUMS` from
this folder.
