"""Proper-scoring, calibration, ranking, and baseline primitives (ml_upgrade_backlog.md Epic L).

This module is the canonical home of count-prediction scoring. It is pure
numpy/pandas so it stays importable in lightweight environments;
``ml_training`` imports from here (never the reverse) so the two can never
form a cycle.

Distribution convention: predictions are Poisson means ``mu`` optionally
paired with a negative-binomial dispersion ``alpha`` (Var = mu + alpha*mu^2,
so ``alpha = 0`` degenerates to Poisson). This matches the two-stage fit in
ticket M1.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_LGAMMA = np.vectorize(math.lgamma, otypes=[float])
_MIN_MU = 1e-12


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)


def poisson_log_loss(y_true: Iterable[float], y_mean: Iterable[float]) -> float:
    """Mean Poisson negative log likelihood, including the log-factorial term."""

    y = _as_float_array(y_true)
    mu = np.clip(_as_float_array(y_mean), _MIN_MU, None)
    return float(np.mean(mu - y * np.log(mu) + _LGAMMA(y + 1.0)))


def count_cdf(
    k_grid: np.ndarray,
    mu: np.ndarray,
    alpha: float = 0.0,
) -> np.ndarray:
    """CDF matrix ``F[i, j] = P(Y_i <= k_grid[j])`` for Poisson/NB means.

    ``alpha`` is the NB dispersion (Var = mu + alpha*mu^2); 0 means Poisson.
    Computed by cumulative summation of the pmf in log space, vectorized over
    a (n_samples, n_grid) grid.
    """

    mu = np.clip(_as_float_array(mu), _MIN_MU, None)[:, None]
    k = np.asarray(k_grid, dtype=float)[None, :]
    if alpha and alpha > 0.0:
        r = 1.0 / float(alpha)
        log_pmf = (
            _LGAMMA(k + r)
            - _LGAMMA(np.full_like(k, r))
            - _LGAMMA(k + 1.0)
            + r * np.log(r / (r + mu))
            + k * np.log(mu / (r + mu))
        )
    else:
        log_pmf = k * np.log(mu) - mu - _LGAMMA(k + 1.0)
    pmf = np.exp(log_pmf)
    return np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)


def prob_at_least(mu: Iterable[float], threshold: int, alpha: float = 0.0) -> np.ndarray:
    """P(Y >= threshold) under Poisson (alpha=0) or NB (alpha>0)."""

    if threshold <= 0:
        return np.ones(len(_as_float_array(mu)))
    grid = np.arange(threshold, dtype=float)
    cdf = count_cdf(grid, _as_float_array(mu), alpha=alpha)
    return np.clip(1.0 - cdf[:, -1], 0.0, 1.0)


def ranked_probability_score(
    y_true: Iterable[float],
    mu: Iterable[float],
    alpha: float = 0.0,
    max_k: int = 15,
) -> float:
    """Discrete ranked probability score (CRPS for counts), lower is better.

    ``rps_i = sum_k (F_i(k) - 1{y_i <= k})^2`` summed over ``k = 0..K`` where
    ``K = max(max_k, observed max + 5)`` so truncation never bites the
    observed support. Proper for count forecasts and, unlike log loss,
    sensitive to dispersion errors — the primary M1/N-series model-selection
    metric.
    """

    y = _as_float_array(y_true)
    mu_arr = _as_float_array(mu)
    if len(y) == 0:
        return float("nan")
    upper = int(max(max_k, (np.nanmax(y) if len(y) else 0) + 5))
    grid = np.arange(upper + 1, dtype=float)
    cdf = count_cdf(grid, mu_arr, alpha=alpha)
    indicator = (y[:, None] <= grid[None, :]).astype(float)
    return float(np.mean(np.sum((cdf - indicator) ** 2, axis=1)))


def threshold_calibration(
    event_occurred: Iterable[bool],
    predicted_probability: Iterable[float],
    n_bins: int = 10,
) -> Dict[str, object]:
    """Brier score, expected calibration error, and the reliability table.

    ``event_occurred`` are outcome indicators (e.g. ``y >= 1``);
    ``predicted_probability`` the model's P(event). The reliability table has
    one row per equal-width probability bin with the bin's mean prediction,
    empirical frequency, and count; ECE is the count-weighted mean absolute
    gap between the two.
    """

    y = _as_float_array(event_occurred)
    p = np.clip(_as_float_array(predicted_probability), 0.0, 1.0)
    if len(y) == 0:
        return {"brier": float("nan"), "ece": float("nan"), "reliability": pd.DataFrame()}

    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins; probability 0 lands in the first bin
    bin_index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted = float(p[mask].mean())
        empirical = float(y[mask].mean())
        ece += (count / len(y)) * abs(mean_predicted - empirical)
        rows.append({
            "bin_lower": float(edges[b]),
            "bin_upper": float(edges[b + 1]),
            "count": count,
            "mean_predicted": mean_predicted,
            "empirical_frequency": empirical,
        })
    return {"brier": brier, "ece": float(ece), "reliability": pd.DataFrame(rows)}


def estimate_nb_alpha(y_true: Iterable[float], mu: Iterable[float]) -> float:
    """Method-of-moments NB dispersion: Var = mu + alpha*mu^2 (backlog M1).

    ``alpha_hat = max(0, sum((y - mu)^2 - mu) / sum(mu^2))`` — zero recovers
    Poisson; the estimator is consistent when the conditional mean is right.
    """

    y = _as_float_array(y_true)
    mu_arr = np.clip(_as_float_array(mu), _MIN_MU, None)
    if len(y) == 0:
        return 0.0
    denominator = float(np.sum(mu_arr ** 2))
    if denominator <= 0.0:
        return 0.0
    numerator = float(np.sum((y - mu_arr) ** 2 - mu_arr))
    return max(0.0, numerator / denominator)


def estimate_team_total_nb_alpha(
    frame: pd.DataFrame,
    *,
    y_true: Iterable[float],
    mu: Iterable[float],
    group_cols: Sequence[str] = ("fixture_id", "team_id"),
) -> float:
    """NB dispersion of TEAM TOTALS: sums y and mu per group, then fits alpha.

    This is the dispersion the fixture simulator uses for its team-total
    draws (backlog §2.4) — team totals are usually more overdispersed than
    player counts because within-team correlation accumulates.
    """

    if not set(group_cols).issubset(frame.columns) or len(frame) == 0:
        return 0.0
    grouped = pd.DataFrame({
        "__y__": _as_float_array(y_true),
        "__mu__": _as_float_array(mu),
    })
    for col in group_cols:
        grouped[col] = frame[col].to_numpy()
    totals = grouped.groupby(list(group_cols))[["__y__", "__mu__"]].sum()
    return estimate_nb_alpha(totals["__y__"], totals["__mu__"])


def dispersion_index(y_true: Iterable[float], mu: Iterable[float]) -> float:
    """Mean squared Pearson residual: ~1 under Poisson, >1 when overdispersed."""

    y = _as_float_array(y_true)
    mu_arr = np.clip(_as_float_array(mu), _MIN_MU, None)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((y - mu_arr) ** 2 / mu_arr))


def _spearman(pred: np.ndarray, actual: np.ndarray) -> Optional[float]:
    """Spearman rho via Pearson correlation of average ranks; None if degenerate."""

    pred_rank = pd.Series(pred).rank(method="average").to_numpy()
    actual_rank = pd.Series(actual).rank(method="average").to_numpy()
    if np.std(pred_rank) == 0.0 or np.std(actual_rank) == 0.0:
        return None
    return float(np.corrcoef(pred_rank, actual_rank)[0, 1])


def within_fixture_ranking(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str,
    group_cols: Sequence[str] = ("fixture_id", "team_id"),
) -> Dict[str, float]:
    """Within-group ranking quality: who leads the team in this event?

    Returns the mean Spearman rho across groups plus leader hit rates:
    ``top1_hit_rate`` — the predicted leader's actual count equals the
    group's actual maximum (ties count as hits); ``top3_hit_rate`` — the
    predicted leader's actual count is within the group's top 3 values.
    Groups whose actuals are all zero cannot rank anyone and are excluded
    (reported via ``groups_skipped_all_zero``); groups with constant
    predictions or actuals are excluded from Spearman only.
    """

    spearmans = []
    top1_hits = []
    top3_hits = []
    skipped_all_zero = 0
    for _, group in frame.groupby(list(group_cols)):
        actual = pd.to_numeric(group[actual_col], errors="coerce").fillna(0.0).to_numpy()
        pred = pd.to_numeric(group[pred_col], errors="coerce").fillna(0.0).to_numpy()
        if len(group) < 2:
            continue
        if not (actual > 0).any():
            skipped_all_zero += 1
            continue
        rho = _spearman(pred, actual)
        if rho is not None:
            spearmans.append(rho)
        leader = int(np.argmax(pred))
        top1_hits.append(float(actual[leader] == actual.max()))
        top3_hits.append(float(actual[leader] >= np.sort(actual)[-min(3, len(actual))]))
    return {
        "spearman_mean": float(np.mean(spearmans)) if spearmans else float("nan"),
        "top1_hit_rate": float(np.mean(top1_hits)) if top1_hits else float("nan"),
        "top3_hit_rate": float(np.mean(top3_hits)) if top3_hits else float("nan"),
        "groups_scored": float(len(top1_hits)),
        "groups_skipped_all_zero": float(skipped_all_zero),
    }


def position_group_rates(
    train_frame: pd.DataFrame,
    target: str,
    *,
    exposure: Iterable[float],
    group_col: str = "position_group",
) -> Dict[str, float]:
    """Per-90 event rates per position group with a ``__global__`` fallback.

    ``rate_g = sum(y_g) / sum(exposure_g)`` in 90-minute exposure units — the
    maximum-likelihood Poisson rate for the segment.
    """

    y = pd.to_numeric(train_frame[target], errors="coerce").fillna(0.0).to_numpy()
    exp_arr = np.clip(_as_float_array(exposure), 1e-6, None)
    groups = (
        train_frame[group_col].fillna("unknown").astype(str).to_numpy()
        if group_col in train_frame.columns
        else np.full(len(train_frame), "unknown")
    )
    rates: Dict[str, float] = {"__global__": float(y.sum() / exp_arr.sum())}
    for group in np.unique(groups):
        mask = groups == group
        rates[str(group)] = float(y[mask].sum() / exp_arr[mask].sum())
    return rates


def position_group_baseline_means(
    eval_frame: pd.DataFrame,
    rates: Mapping[str, float],
    *,
    exposure: Iterable[float],
    group_col: str = "position_group",
) -> np.ndarray:
    """Baseline means: position-group per-90 rate x exposure."""

    exp_arr = np.clip(_as_float_array(exposure), 1e-6, None)
    groups = (
        eval_frame[group_col].fillna("unknown").astype(str).to_numpy()
        if group_col in eval_frame.columns
        else np.full(len(eval_frame), "unknown")
    )
    fallback = rates.get("__global__", 0.0)
    rate_values = np.array([rates.get(group, fallback) for group in groups])
    return rate_values * exp_arr


def player_l5_baseline_means(
    eval_frame: pd.DataFrame,
    target: str,
    *,
    exposure: Iterable[float],
    fallback_means: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Baseline means from the player's own L5 per-90 rate.

    Rows without usable history (missing or zero ``<target>_l5_p90``) fall
    back to ``fallback_means`` (typically the position-group baseline) so the
    baseline never emits a structurally impossible hard zero.
    """

    exp_arr = np.clip(_as_float_array(exposure), 1e-6, None)
    column = f"{target}_l5_p90"
    if column in eval_frame.columns:
        rate = pd.to_numeric(eval_frame[column], errors="coerce").fillna(0.0).to_numpy()
    else:
        rate = np.zeros(len(eval_frame))
    means = rate * exp_arr
    if fallback_means is not None:
        means = np.where(rate > 0.0, means, _as_float_array(fallback_means))
    return means


def skill_score(model_loss: float, baseline_loss: float) -> float:
    """1 - model/baseline for a lower-is-better loss; positive beats baseline."""

    if not np.isfinite(model_loss) or not np.isfinite(baseline_loss) or abs(baseline_loss) < 1e-12:
        return float("nan")
    return float(1.0 - model_loss / baseline_loss)


def rolling_origin_folds(
    frame: pd.DataFrame,
    *,
    season_column: str = "league_season",
    min_train_seasons: int = 2,
) -> Iterable[Tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Expanding-window chronological folds: train < season N, validate = N.

    Yields ``(season, train_df, valid_df)`` for every season with at least
    ``min_train_seasons`` earlier seasons in the frame — the leakage-safe
    harness every Epic M adoption decision must run under.
    """

    if season_column not in frame.columns:
        raise ValueError(f"Missing season column: {season_column}")
    seasons = pd.to_numeric(frame[season_column], errors="coerce")
    unique_seasons = sorted(int(season) for season in seasons.dropna().unique())
    for position, season in enumerate(unique_seasons):
        if position < min_train_seasons:
            continue
        train = frame[seasons < season].copy()
        valid = frame[seasons == season].copy()
        if train.empty or valid.empty:
            continue
        yield season, train, valid


def validation_metric_suite(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    y_valid: Iterable[float],
    mu_valid: Iterable[float],
    target: str,
    train_exposure: Iterable[float],
    valid_exposure: Iterable[float],
    alpha: float = 0.0,
    thresholds: Sequence[int] = (1, 2),
) -> Dict[str, object]:
    """The full L-series metric suite for one target's validation slice.

    Returns ``{"metrics": {name: float}, "artifacts": {name: DataFrame}}``.
    Scalars are quota-safe to log to MLflow; artifacts (reliability tables)
    belong in run artifacts (Working Agreement #4).
    """

    y = _as_float_array(y_valid)
    mu = _as_float_array(mu_valid)

    metrics: Dict[str, float] = {
        "rps": ranked_probability_score(y, mu, alpha=alpha),
        "dispersion_index": dispersion_index(y, mu),
    }
    artifacts: Dict[str, pd.DataFrame] = {}

    for threshold in thresholds:
        calibration = threshold_calibration(y >= threshold, prob_at_least(mu, threshold, alpha=alpha))
        metrics[f"brier_ge_{threshold}"] = calibration["brier"]
        metrics[f"ece_ge_{threshold}"] = calibration["ece"]
        artifacts[f"reliability_ge_{threshold}"] = calibration["reliability"]

    model_logloss = poisson_log_loss(y, mu)
    rates = position_group_rates(train_frame, target, exposure=train_exposure)
    posgroup_mu = position_group_baseline_means(valid_frame, rates, exposure=valid_exposure)
    player_mu = player_l5_baseline_means(
        valid_frame, target, exposure=valid_exposure, fallback_means=posgroup_mu
    )
    metrics["skill_vs_posgroup_rate"] = skill_score(model_logloss, poisson_log_loss(y, posgroup_mu))
    metrics["skill_vs_player_l5"] = skill_score(model_logloss, poisson_log_loss(y, player_mu))

    if {"fixture_id", "team_id"}.issubset(valid_frame.columns):
        ranking_frame = valid_frame[["fixture_id", "team_id"]].copy()
        ranking_frame["__pred__"] = mu
        ranking_frame["__actual__"] = y
        ranking = within_fixture_ranking(
            ranking_frame, pred_col="__pred__", actual_col="__actual__"
        )
        metrics["rank_spearman"] = ranking["spearman_mean"]
        metrics["top1_hit_rate"] = ranking["top1_hit_rate"]
        metrics["top3_hit_rate"] = ranking["top3_hit_rate"]

    return {"metrics": metrics, "artifacts": artifacts}
