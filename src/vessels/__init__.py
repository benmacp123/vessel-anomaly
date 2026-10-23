"""Vessel-behaviour anomaly detection from public AIS tracks.

Subpackages, each depending only on the core and itself, except
``evaluation``, which may use them all:

- ``core``: cleaning, segmentation and detectors over polars frames (no I/O).
- ``io``: reading raw sources into the internal schema, and the extracts.
- ``sim``: arrival simulation and synthetic injection.
- ``stream``: micro-batch wrapper with per-segment state and a watermark.
- ``evaluation``: frozen-config scoring, alert rates and figures.
"""
