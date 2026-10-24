"""Tests for the realism check's AUC and discriminator on known cases."""

import numpy as np
import polars as pl
import pytest

from vessels.evaluation.realism import FEATURES, auc, discriminator, distributions


def test_auc_hand_built():
    """AUC matches a hand count of correctly ordered pairs, ties halved."""
    scores = np.array([0.9, 0.8, 0.4, 0.4, 0.1])
    labels = np.array([1, 0, 1, 0, 0])
    # Planted 0.9 beats all 3 real; planted 0.4 beats 0.1, ties 0.4, loses
    # to 0.8: (3 + 1.5) / 6.
    assert auc(scores, labels) == pytest.approx(4.5 / 6)


def test_auc_needs_both_classes():
    """With one class only, AUC is undefined."""
    assert np.isnan(auc(np.array([0.1, 0.2]), np.array([1, 1])))


def sample(rng, n, shift):
    """Random feature rows, shifted on every feature.

    Args:
        rng: Random source.
        n: Rows.
        shift: Added to every feature.

    Returns:
        pl.DataFrame: ``FEATURES`` columns.
    """
    return pl.DataFrame({name: rng.normal(shift, 1.0, n) for name in FEATURES})


def test_discriminator_separates_shifted_classes():
    """Clearly different classes give an AUC near 1."""
    rng = np.random.default_rng(0)
    result = discriminator(sample(rng, 300, 2.0), sample(rng, 300, 0.0), seed=1)
    assert result["auc"] > 0.95


def test_discriminator_cannot_separate_same_classes():
    """Identically distributed classes give an AUC near 0.5."""
    rng = np.random.default_rng(0)
    result = discriminator(sample(rng, 400, 0.0), sample(rng, 400, 0.0), seed=1)
    assert abs(result["auc"] - 0.5) < 0.08


def test_distributions_report_every_feature():
    """Each feature gets a row with its KS statistic."""
    rng = np.random.default_rng(0)
    table = distributions(sample(rng, 50, 1.0), sample(rng, 50, 0.0))
    assert table["feature"].to_list() == list(FEATURES)
    assert (table["ks"] > 0.2).all()
