import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from football_analytics.ml_training import (
    PoissonLightGBMConfig,
    run_rolling_origin_backtest,
)


def _synthetic_seasons(seasons=(2021, 2022, 2023, 2024), rows_per_season=240, seed=5):
    """Players whose shot rate depends on one informative feature."""

    rng = np.random.default_rng(seed)
    frames = []
    for season in seasons:
        strength = rng.uniform(0.0, 1.0, size=rows_per_season)
        minutes = rng.uniform(45.0, 95.0, size=rows_per_season)
        rate_per_90 = 0.5 + 2.0 * strength
        shots = rng.poisson(rate_per_90 * minutes / 90.0)
        frames.append(pd.DataFrame({
            "league_season": season,
            "fixture_id": np.repeat(np.arange(rows_per_season // 20) + season * 100, 20),
            "team_id": np.tile([1, 2], rows_per_season // 2),
            "position_group": rng.choice(["F", "M", "D"], size=rows_per_season),
            "player_strength": strength,
            "games_minutes": minutes,
            "shots_total": shots.astype(float),
            "shots_total_l5_p90": rate_per_90 + rng.normal(0, 0.2, size=rows_per_season),
        }))
    return pd.concat(frames, ignore_index=True)


def _fast_config():
    return PoissonLightGBMConfig(
        learning_rate=0.2,
        num_leaves=7,
        min_child_samples=10,
        num_boost_round=40,
        early_stopping_rounds=10,
    )


def test_rolling_origin_backtest_scores_each_fold_with_the_metric_suite():
    frame = _synthetic_seasons()
    result = run_rolling_origin_backtest(
        frame,
        target_columns=["shots_total"],
        feature_columns=["player_strength"],
        config=_fast_config(),
        min_train_seasons=2,
    )

    report = result["report"]
    assert set(report["season"]) == {2023, 2024}
    expected_metrics = {
        "poisson_logloss", "mae", "rmse", "rps", "dispersion_index",
        "brier_ge_1", "ece_ge_1", "skill_vs_posgroup_rate",
        "skill_vs_player_l5", "rank_spearman", "top1_hit_rate",
    }
    assert expected_metrics.issubset(set(report["metric"]))

    summary = result["summary"]
    rps_row = summary[(summary["target"] == "shots_total") & (summary["metric"] == "rps")]
    assert rps_row["folds"].iloc[0] == 2

    # The model sees the true signal, so it must beat the position-group
    # baseline on both folds.
    skill = report[(report["metric"] == "skill_vs_posgroup_rate")]["value"]
    assert (skill > 0.0).all()

    assert result["skipped"].empty


def test_baseline_reference_scores_both_baselines_per_fold():
    from football_analytics.ml_training import run_baseline_reference

    frame = _synthetic_seasons()
    result = run_baseline_reference(
        frame,
        target_columns=["shots_total"],
        min_train_seasons=2,
    )

    report = result["report"]
    assert set(report["baseline"]) == {"posgroup_rate", "player_l5"}
    assert set(report["season"]) == {2023, 2024}
    assert {"poisson_logloss", "rps", "dispersion_index"}.issubset(set(report["metric"]))

    # The player-L5 baseline tracks the true per-player rate, so it must beat
    # the position-group rate baseline on log loss in this synthetic world.
    summary = result["summary"]
    logloss = summary[summary["metric"] == "poisson_logloss"].set_index("baseline")["mean"]
    assert logloss["player_l5"] < logloss["posgroup_rate"]


def test_rolling_origin_backtest_skips_unlearnable_targets():
    frame = _synthetic_seasons()
    frame["cards_red"] = 0.0
    result = run_rolling_origin_backtest(
        frame,
        target_columns=["cards_red"],
        feature_columns=["player_strength"],
        config=_fast_config(),
        min_train_seasons=2,
    )

    assert result["report"].empty
    assert (result["skipped"]["reason"] == "no positive labels in training fold").all()
    assert set(result["skipped"]["season"]) == {2023, 2024}
