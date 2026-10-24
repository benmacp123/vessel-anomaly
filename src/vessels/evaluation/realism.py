"""How far planted gaps look like real ones (PLAN.md, evaluation item 9).

No published protocol validates injected AIS gaps, so two checks are
reported (design reasoning, decided 2026-10-24):

- per-feature distributions of planted and real gaps, with the two-sample
  Kolmogorov-Smirnov statistic;
- a small discriminator: L2-regularised logistic regression on the same
  features, trained on a fixed half of the gaps and scored by AUC on the
  other half. An AUC near 0.5 means the features cannot tell planted gaps
  from real ones; near 1 means planted gaps are easy to spot, and recall
  measured on them says less about real gaps.

Real gaps are those in the planted population: over 6 hours (up to the
planted maximum), both ends covered and away from anchorages. Planted gaps
are the detector's gap rows that overlap a planted positive, so both sides
carry the same measured features.
"""

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import ks_2samp, rankdata

# Features of a gap row, as the discriminator sees them.
FEATURES = ("log_hours", "log_prior_positions", "implied_kn", "log_density")
# Strength of the L2 penalty on the standardised weights.
PENALTY = 1.0


def features(gaps: pl.DataFrame, density: pl.DataFrame) -> pl.DataFrame:
    """Discriminator features for gap rows.

    Args:
        gaps: ``gap_events`` rows.
        density: ``h3_cell``, ``transceiver`` and ``density`` (positions per
            day in the tuning grid).

    Returns:
        pl.DataFrame: One row per gap with ``FEATURES``.
    """
    return (
        gaps.join(
            density,
            left_on=["start_cell", "transceiver"],
            right_on=["h3_cell", "transceiver"],
            how="left",
            maintain_order="left",
        )
        .with_columns(pl.col("density").fill_null(0.0))
        .select(
            log_hours=pl.col("hours").log(),
            log_prior_positions=pl.col("prior_positions").cast(pl.Float64).log1p(),
            implied_kn=pl.col("implied_kn"),
            log_density=pl.col("density").log1p(),
        )
    )


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney form, ties averaged).

    Args:
        scores: Higher means more likely planted.
        labels: 1 for planted, 0 for real.

    Returns:
        float: AUC; NaN when either class is missing.
    """
    n1 = int(labels.sum())
    n0 = labels.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic(x: np.ndarray, y: np.ndarray, penalty: float = PENALTY) -> np.ndarray:
    """L2-regularised logistic regression weights (intercept first).

    Args:
        x: Standardised features, one row per sample.
        y: 0 or 1 labels.
        penalty: L2 strength on the non-intercept weights.

    Returns:
        np.ndarray: Weights, length ``x.shape[1] + 1``.
    """
    design = np.column_stack([np.ones(len(x)), x])

    def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
        """Mean log-loss plus penalty, and its gradient.

        Args:
            w: Weights.

        Returns:
            tuple[float, np.ndarray]: Loss and gradient.
        """
        z = design @ w
        p = 1 / (1 + np.exp(-z))
        value = np.mean(np.logaddexp(0, z) - y * z) + penalty * (
            w[1:] @ w[1:]
        ) / 2 / len(y)
        grad = design.T @ (p - y) / len(y)
        grad[1:] += penalty * w[1:] / len(y)
        return value, grad

    return minimize(loss, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B").x


def discriminator(
    planted: pl.DataFrame, real: pl.DataFrame, seed: int
) -> dict[str, float | int]:
    """Train on a fixed half and report the held-back half's AUC.

    Args:
        planted: ``features`` of planted gaps.
        real: ``features`` of real gaps.
        seed: Seed of the fixed split.

    Returns:
        dict[str, float | int]: ``planted``, ``real``, ``test`` (held-back
        size) and ``auc``.
    """
    x = np.vstack([planted.to_numpy(), real.to_numpy()]).astype(float)
    y = np.concatenate([np.ones(planted.height), np.zeros(real.height)])
    test = np.random.default_rng(seed).random(len(y)) < 0.5
    mean, std = x[~test].mean(axis=0), x[~test].std(axis=0)
    std[std == 0] = 1.0
    weights = fit_logistic((x[~test] - mean) / std, y[~test])
    scores = np.column_stack([np.ones(test.sum()), (x[test] - mean) / std]) @ weights
    return {
        "planted": planted.height,
        "real": real.height,
        "test": int(test.sum()),
        "auc": round(auc(scores, y[test]), 3),
    }


def distributions(planted: pl.DataFrame, real: pl.DataFrame) -> pl.DataFrame:
    """Median, quartiles and KS statistic per feature.

    Args:
        planted: ``features`` of planted gaps.
        real: ``features`` of real gaps.

    Returns:
        pl.DataFrame: One row per feature.
    """
    rows = []
    for name in FEATURES:
        a, b = planted[name].to_numpy(), real[name].to_numpy()
        test = ks_2samp(a, b)
        rows.append(
            {
                "feature": name,
                "planted_q25": float(np.quantile(a, 0.25)),
                "planted_median": float(np.median(a)),
                "planted_q75": float(np.quantile(a, 0.75)),
                "real_q25": float(np.quantile(b, 0.25)),
                "real_median": float(np.median(b)),
                "real_q75": float(np.quantile(b, 0.75)),
                "ks": float(test.statistic),
                "ks_p": float(test.pvalue),
            }
        )
    return pl.DataFrame(rows).with_columns(pl.selectors.float().round(3))
