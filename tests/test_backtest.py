import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from football_analytics.ml_training import (
    PoissonLightGBMConfig,
    degrade_rows_to_serving_shape,
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


def test_cross_fold_rps_prefers_the_informative_config():
    from football_analytics.ml_training import cross_fold_rps

    frame = _synthetic_seasons()
    good = cross_fold_rps(
        frame,
        target="shots_total",
        feature_columns=["player_strength"],
        config=_fast_config(),
    )
    # A config that cannot learn (zero boosting rounds is invalid, so use a
    # single stump round with tiny learning rate) must score worse.
    weak = cross_fold_rps(
        frame,
        target="shots_total",
        feature_columns=["player_strength"],
        config=PoissonLightGBMConfig(
            learning_rate=0.001,
            num_leaves=2,
            min_child_samples=200,
            num_boost_round=1,
            early_stopping_rounds=0,
        ),
    )
    assert np.isfinite(good) and np.isfinite(weak)
    assert good < weak

    # Unlearnable target -> inf, so tuners reject the trial.
    frame["cards_red"] = 0.0
    assert cross_fold_rps(
        frame,
        target="cards_red",
        feature_columns=["player_strength"],
        config=_fast_config(),
    ) == float("inf")


def test_load_tuned_configs_applies_the_adoption_gate(tmp_path):
    import json

    from football_analytics.ml_training import load_tuned_configs

    adopted = {
        "target": "shots_total",
        "params": {"learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 20,
                   "feature_fraction": 0.9, "bagging_fraction": 0.8, "lambda_l2": 0.5},
        "num_boost_round": 1200,
        "early_stopping_rounds": 50,
        "adopted": True,
    }
    rejected = {"target": "cards_red", "params": {"learning_rate": 0.2}, "adopted": False}
    (tmp_path / "shots_total.json").write_text(json.dumps(adopted), encoding="utf-8")
    (tmp_path / "cards_red.json").write_text(json.dumps(rejected), encoding="utf-8")
    (tmp_path / "garbage.json").write_text("{not json", encoding="utf-8")

    configs = load_tuned_configs(tmp_path)
    assert set(configs) == {"shots_total"}
    assert configs["shots_total"].learning_rate == pytest.approx(0.1)
    assert configs["shots_total"].num_leaves == 15
    assert configs["shots_total"].num_boost_round == 1200

    assert load_tuned_configs(tmp_path / "missing") == {}


def _with_consistent_elo_family(frame):
    """Elo family where fixture-exact values equal the serving recompute."""

    out = frame.copy()
    out["team_elo_attack_pre"] = 0.25
    out["team_elo_defense_pre"] = 0.125
    out["opponent_elo_attack_pre"] = 0.1
    out["opponent_elo_defense_pre"] = 0.05
    out["player_offensive_modifier_pre"] = out["player_strength"]
    out["player_offensive_elo_pre"] = 0.25 + out["player_strength"]
    out["player_offensive_rating_pre"] = out["player_offensive_elo_pre"]
    out["player_defensive_modifier_pre"] = 0.0
    out["player_defensive_elo_pre"] = 0.125
    out["player_defensive_rating_pre"] = 0.125
    out["missed_fixture_count_pre"] = 0
    return out


def test_degrade_rows_to_serving_shape_mirrors_mart_inference_coalesces():
    # FA-102: the helper must reproduce the mart's inference-row semantics —
    # modifiers/missed count pass through (coalesced to 0 for players without
    # Elo history), elo/rating are rebuilt as team baseline + modifier, and
    # the derived interactions follow the rebuilt bases. Labels stay intact.
    frame = pd.DataFrame([
        {
            "team_elo_attack_pre": 0.25,
            "team_elo_defense_pre": 0.125,
            "opponent_elo_attack_pre": 0.1,
            "opponent_elo_defense_pre": 0.05,
            "player_offensive_modifier_pre": 0.5,
            "player_defensive_modifier_pre": 0.25,
            # fixture-exact values deliberately inconsistent so the test
            # proves the recompute wins over pass-through
            "player_offensive_elo_pre": -9.0,
            "player_defensive_elo_pre": -9.0,
            "player_offensive_rating_pre": -9.0,
            "player_defensive_rating_pre": -9.0,
            "missed_fixture_count_pre": 2,
            "shots_total": 1.0,
        },
        {
            # player unseen by Elo history: the mart coalesces modifiers to 0
            # and elo/rating to the team baseline
            "team_elo_attack_pre": 0.25,
            "team_elo_defense_pre": 0.125,
            "opponent_elo_attack_pre": 0.1,
            "opponent_elo_defense_pre": 0.05,
            "player_offensive_modifier_pre": None,
            "player_defensive_modifier_pre": None,
            "player_offensive_elo_pre": None,
            "player_defensive_elo_pre": None,
            "player_offensive_rating_pre": None,
            "player_defensive_rating_pre": None,
            "missed_fixture_count_pre": None,
            "shots_total": 0.0,
        },
    ])

    degraded = degrade_rows_to_serving_shape(frame)
    row = degraded.iloc[0]

    assert row["player_offensive_elo_pre"] == pytest.approx(0.75)
    assert row["player_offensive_rating_pre"] == pytest.approx(0.75)
    assert row["player_defensive_elo_pre"] == pytest.approx(0.375)
    assert row["player_offensive_modifier_pre"] == pytest.approx(0.5)
    assert row["missed_fixture_count_pre"] == 2
    assert row["shots_total"] == 1.0
    assert row["player_attack_delta_vs_team"] == pytest.approx(0.5)
    assert row["player_attack_vs_opp_defense"] == pytest.approx(0.70)

    fallback = degraded.iloc[1]
    assert fallback["player_offensive_modifier_pre"] == 0.0
    assert fallback["player_offensive_elo_pre"] == pytest.approx(0.25)
    assert fallback["player_defensive_elo_pre"] == pytest.approx(0.125)
    assert fallback["missed_fixture_count_pre"] == 0

    # input frame untouched
    assert frame.iloc[0]["player_offensive_elo_pre"] == -9.0


def test_serving_parity_backtest_reports_zero_delta_on_consistent_features():
    # The standing FA-102 regression gate: post-FA-101 the serving reshape is
    # an identity on self-consistent rows, so every paired metric delta ~ 0.
    frame = _with_consistent_elo_family(_synthetic_seasons())

    result = run_rolling_origin_backtest(
        frame,
        target_columns=["shots_total"],
        feature_columns=["player_offensive_elo_pre", "player_offensive_modifier_pre"],
        config=_fast_config(),
        min_train_seasons=2,
        serving_parity=True,
    )

    parity = result["parity_report"]
    assert not parity.empty
    finite = parity[np.isfinite(parity["delta"])]
    assert not finite.empty
    assert (finite["delta"].abs() < 1e-9).all()

    summary = result["parity_summary"]
    assert {"training_shaped", "serving_shaped", "delta", "folds"}.issubset(summary.columns)
    rps = summary[summary["metric"] == "rps"]
    assert rps["folds"].iloc[0] == 2


def test_serving_parity_backtest_measures_skew_when_serving_cannot_reproduce_features():
    # E-5 replica: the model leans on a fixture-exact rating whose signal the
    # serving path cannot rebuild (modifier carries nothing), so the
    # serving-shaped fold must score strictly worse — the paired delta is the
    # measured cost of the skew.
    frame = _synthetic_seasons()
    frame["team_elo_attack_pre"] = 0.25
    frame["team_elo_defense_pre"] = 0.125
    frame["opponent_elo_attack_pre"] = 0.1
    frame["opponent_elo_defense_pre"] = 0.05
    frame["player_offensive_modifier_pre"] = 0.0
    frame["player_offensive_rating_pre"] = 0.25 + frame["player_strength"]
    frame["player_offensive_elo_pre"] = frame["player_offensive_rating_pre"]
    frame["player_defensive_modifier_pre"] = 0.0
    frame["player_defensive_elo_pre"] = 0.125
    frame["player_defensive_rating_pre"] = 0.125
    frame["missed_fixture_count_pre"] = 0

    result = run_rolling_origin_backtest(
        frame,
        target_columns=["shots_total"],
        feature_columns=["player_offensive_rating_pre"],
        config=_fast_config(),
        min_train_seasons=2,
        serving_parity=True,
    )

    parity = result["parity_report"]
    rps_delta = parity[parity["metric"] == "rps"]["delta"]
    assert (rps_delta > 0.0).all()


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
