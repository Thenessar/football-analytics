import math

import pandas as pd
import pytest

from football_analytics.elo import (
    assemble_hierarchical_feature_frame,
    assert_no_unhandled_event_nulls,
    build_player_elo_history,
    build_team_elo_history,
    coalesce_structural_event_zeros,
)
from football_analytics.ml_training import (
    DEFAULT_LIGHTGBM_FEATURES,
    _log_model_kwargs,
    _prepare_xy,
    add_model_interaction_features,
    count_threshold_probabilities,
    evaluate_count_predictions,
    poisson_log_loss,
    temporal_train_validation_split,
)
from scripts.train_poisson_lgbm import REQUIRED_ELO_FEATURE_COLUMNS, build_training_feature_frame


def _fixture(fixture_id, date, home_goals=1, away_goals=0):
    return {
        "fixture_id": fixture_id,
        "fixture_date_utc": date,
        "home_team_id": 1,
        "home_team_name": "Team A",
        "away_team_id": 2,
        "away_team_name": "Team B",
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


def test_team_elo_snapshots_are_pre_match_and_chronological():
    fixtures = pd.DataFrame([
        _fixture(1, "2023-01-01T12:00:00Z", home_goals=2, away_goals=0),
        _fixture(2, "2023-01-08T12:00:00Z", home_goals=0, away_goals=0),
    ])
    rankings = pd.DataFrame([
        {"team_name": "Team A", "rating": 1600.0},
        {"team_name": "Team B", "rating": 1500.0},
    ])

    history = build_team_elo_history(fixtures, rankings)
    first_home = history[(history["fixture_id"] == 1) & (history["team_id"] == 1)].iloc[0]
    second_home = history[(history["fixture_id"] == 2) & (history["team_id"] == 1)].iloc[0]

    assert first_home["team_elo_general_pre"] == 1550.0
    assert first_home["team_elo_attack_pre"] == 0.0
    assert first_home["team_elo_defense_pre"] == 0.0

    assert second_home["team_elo_general_pre"] > first_home["team_elo_general_pre"]
    assert second_home["team_elo_attack_pre"] > first_home["team_elo_attack_pre"]
    assert second_home["expected_goals_for_pre"] > first_home["expected_goals_for_pre"]
    assert first_home["team_elo_attack_post"] > first_home["team_elo_attack_pre"]
    assert first_home["team_elo_defense_post"] > first_home["team_elo_defense_pre"]


def test_player_modifier_decays_toward_current_team_baseline_after_inactivity():
    fixtures = pd.DataFrame([
        _fixture(i, f"2023-01-{i:02d}T12:00:00Z", home_goals=1, away_goals=0)
        for i in range(1, 13)
    ])
    team_history = build_team_elo_history(fixtures)
    appearances = []

    appearances.extend([
        {
            "fixture_id": 1,
            "fixture_date_utc": "2023-01-01T12:00:00Z",
            "team_id": 1,
            "team_name": "Team A",
            "player_id": 10,
            "player_name": "Explosive Winger",
            "games_minutes": 90,
            "shots_total": 5,
            "shots_on": 2,
            "dribbles_attempts": 4,
            "goals_assists": 0,
            "tackles_interceptions": 0,
            "tackles_total": 0,
            "fouls_committed": 0,
        },
        {
            "fixture_id": 1,
            "fixture_date_utc": "2023-01-01T12:00:00Z",
            "team_id": 1,
            "team_name": "Team A",
            "player_id": 11,
            "player_name": "Control Midfielder",
            "games_minutes": 90,
            "shots_total": 0,
            "shots_on": 0,
            "dribbles_attempts": 0,
            "goals_assists": 0,
            "tackles_interceptions": 0,
            "tackles_total": 0,
            "fouls_committed": 0,
        },
    ])

    for fixture_id in range(2, 12):
        appearances.append({
            "fixture_id": fixture_id,
            "fixture_date_utc": f"2023-01-{fixture_id:02d}T12:00:00Z",
            "team_id": 1,
            "team_name": "Team A",
            "player_id": 11,
            "player_name": "Control Midfielder",
            "games_minutes": 90,
            "shots_total": 0,
            "shots_on": 0,
            "dribbles_attempts": 0,
            "goals_assists": 0,
            "tackles_interceptions": 0,
            "tackles_total": 0,
            "fouls_committed": 0,
        })

    appearances.extend([
        {
            "fixture_id": 12,
            "fixture_date_utc": "2023-01-12T12:00:00Z",
            "team_id": 1,
            "team_name": "Team A",
            "player_id": 10,
            "player_name": "Explosive Winger",
            "games_minutes": 90,
            "shots_total": 1,
            "shots_on": 0,
            "dribbles_attempts": 1,
            "goals_assists": 0,
            "tackles_interceptions": 0,
            "tackles_total": 0,
            "fouls_committed": 0,
        },
        {
            "fixture_id": 12,
            "fixture_date_utc": "2023-01-12T12:00:00Z",
            "team_id": 1,
            "team_name": "Team A",
            "player_id": 99,
            "player_name": "New Callup",
            "games_minutes": 90,
            "shots_total": 0,
            "shots_on": 0,
            "dribbles_attempts": 0,
            "goals_assists": 0,
            "tackles_interceptions": 0,
            "tackles_total": 0,
            "fouls_committed": 0,
        },
    ])

    player_history = build_player_elo_history(pd.DataFrame(appearances), team_history)

    winger_return = player_history[
        (player_history["fixture_id"] == 12)
        & (player_history["player_id"] == 10)
    ].iloc[0]
    new_callup = player_history[
        (player_history["fixture_id"] == 12)
        & (player_history["player_id"] == 99)
    ].iloc[0]

    post_first_match_modifier = 0.05 * ((5 + 0.5 * 2 + 0.35 * 4) - ((5 + 0.5 * 2 + 0.35 * 4) / 2))
    expected_decayed_modifier = post_first_match_modifier * (0.85 ** 10)

    assert winger_return["missed_fixture_count_pre"] == 10
    assert winger_return["player_offensive_modifier_pre"] == pytest.approx(expected_decayed_modifier)
    assert abs(winger_return["player_offensive_modifier_pre"]) < 0.05
    assert new_callup["missed_fixture_count_pre"] == 0
    assert new_callup["player_offensive_modifier_pre"] == 0.0
    assert new_callup["player_offensive_rating_pre"] == new_callup["team_elo_attack_pre"]


def test_zero_minute_player_snapshot_decays_modifier_before_emit():
    fixtures = pd.DataFrame([
        _fixture(1, "2023-01-01T12:00:00Z", home_goals=1, away_goals=0),
        _fixture(2, "2023-01-02T12:00:00Z", home_goals=1, away_goals=0),
    ])
    team_history = build_team_elo_history(fixtures)
    appearances = pd.DataFrame([
        {
            "fixture_id": 1,
            "fixture_date_utc": "2023-01-01T12:00:00Z",
            "team_id": 1,
            "team_name": "Team A",
            "player_id": 10,
            "player_name": "Explosive Winger",
            "games_minutes": 90,
            "shots_total": 5,
            "shots_on": 2,
            "dribbles_attempts": 4,
            "goals_assists": 0,
            "tackles_interceptions": 0,
            "tackles_total": 0,
            "fouls_committed": 0,
        },
        {
            "fixture_id": 1,
            "fixture_date_utc": "2023-01-01T12:00:00Z",
            "team_id": 1,
            "team_name": "Team A",
            "player_id": 11,
            "player_name": "Control Midfielder",
            "games_minutes": 90,
            "shots_total": 0,
            "shots_on": 0,
            "dribbles_attempts": 0,
            "goals_assists": 0,
            "tackles_interceptions": 0,
            "tackles_total": 0,
            "fouls_committed": 0,
        },
        {
            "fixture_id": 2,
            "fixture_date_utc": "2023-01-02T12:00:00Z",
            "team_id": 1,
            "team_name": "Team A",
            "player_id": 10,
            "player_name": "Explosive Winger",
            "games_minutes": 0,
            "shots_total": None,
            "shots_on": None,
            "dribbles_attempts": None,
            "goals_assists": None,
            "tackles_interceptions": None,
            "tackles_total": None,
            "fouls_committed": None,
        },
        {
            "fixture_id": 2,
            "fixture_date_utc": "2023-01-02T12:00:00Z",
            "team_id": 1,
            "team_name": "Team A",
            "player_id": 11,
            "player_name": "Control Midfielder",
            "games_minutes": 90,
            "shots_total": 0,
            "shots_on": 0,
            "dribbles_attempts": 0,
            "goals_assists": 0,
            "tackles_interceptions": 0,
            "tackles_total": 0,
            "fouls_committed": 0,
        },
    ])

    player_history = build_player_elo_history(appearances, team_history)
    inactive_snapshot = player_history[
        (player_history["fixture_id"] == 2)
        & (player_history["player_id"] == 10)
    ].iloc[0]

    post_first_match_modifier = 0.05 * ((5 + 0.5 * 2 + 0.35 * 4) - ((5 + 0.5 * 2 + 0.35 * 4) / 2))

    assert inactive_snapshot["games_minutes"] == 0.0
    assert inactive_snapshot["missed_fixture_count_pre"] == 1
    assert inactive_snapshot["player_offensive_modifier_pre"] == pytest.approx(
        post_first_match_modifier * 0.85
    )
    assert inactive_snapshot["player_offensive_elo_pre"] == pytest.approx(
        inactive_snapshot["team_elo_attack_pre"] + inactive_snapshot["player_offensive_modifier_pre"]
    )


def test_structural_zero_imputation_and_null_guard():
    frame = pd.DataFrame({
        "shots_total": [None, 2],
        "offsides": [None, None],
        "tackles_interceptions": [1, None],
    })

    zeroed = coalesce_structural_event_zeros(frame)

    assert zeroed["shots_total"].tolist() == [0.0, 2.0]
    assert zeroed["offsides"].tolist() == [0.0, 0.0]
    assert zeroed["tackles_interceptions"].tolist() == [1.0, 0.0]
    assert_no_unhandled_event_nulls(zeroed)

    with pytest.raises(ValueError, match="shots_total"):
        assert_no_unhandled_event_nulls(frame)


def test_poisson_metrics_and_threshold_distribution_are_count_safe():
    metrics = evaluate_count_predictions([0, 2], [0.25, 1.75])
    distribution = count_threshold_probabilities(1.64, thresholds=(1, 2))

    assert poisson_log_loss([0, 2], [0.25, 1.75]) == metrics["poisson_logloss"]
    assert metrics["mae"] == pytest.approx(0.25)
    assert metrics["rmse"] == pytest.approx(0.25)
    assert distribution["mean"] == 1.64
    assert distribution["p_ge_1"] == pytest.approx(1.0 - math.exp(-1.64))
    assert 0.0 < distribution["p_ge_2"] < distribution["p_ge_1"] < 1.0


def test_prepare_xy_allows_exposure_column_as_model_feature():
    frame = pd.DataFrame({
        "shots_total": [0, 2],
        "games_minutes": [45, 90],
        "is_starter": [0, 1],
    })

    X, y, exposure = _prepare_xy(
        frame,
        target_column="shots_total",
        feature_columns=["games_minutes", "is_starter", "games_minutes"],
        exposure_column="games_minutes",
    )

    assert X.columns.tolist() == ["games_minutes", "is_starter"]
    assert y.tolist() == [0.0, 2.0]
    assert exposure.tolist() == [0.5, 1.0]


def test_training_script_requires_dbt_materialized_elo_columns():
    feature_frame = pd.DataFrame([{
        "fixture_id": 1,
        "fixture_date_utc": "2023-01-01T12:00:00Z",
        "team_id": 1,
        "player_id": 10,
        "games_minutes": 90,
    }])

    with pytest.raises(ValueError, match="dbt-materialized ELO"):
        build_training_feature_frame(feature_frame)

    enriched_feature_frame = feature_frame.assign(**{
        "team_elo_general_pre": 1510.0,
        "opponent_elo_general_pre": 1490.0,
        "team_elo_attack_pre": 0.2,
        "team_elo_defense_pre": 0.1,
        "opponent_elo_attack_pre": -0.1,
        "opponent_elo_defense_pre": 0.05,
        "expected_goals_for_pre": 1.4,
        "expected_goals_against_pre": 1.0,
        "player_offensive_modifier_pre": 0.3,
        "player_defensive_modifier_pre": -0.2,
        "player_offensive_elo_pre": 0.5,
        "player_defensive_elo_pre": -0.1,
        "player_offensive_rating_pre": 0.5,
        "player_defensive_rating_pre": -0.1,
        "missed_fixture_count_pre": 0,
        "team_lineup_attack_strength": 0.25,
        "team_lineup_defense_strength": 0.05,
    })
    training_frame = build_training_feature_frame(enriched_feature_frame)

    assert training_frame.iloc[0]["team_elo_general_diff"] == 20.0
    assert training_frame.iloc[0]["player_attack_vs_opp_defense"] == pytest.approx(0.45)
    assert training_frame.iloc[0]["lineup_attack_delta_vs_team"] == pytest.approx(0.05)


def test_model_interaction_features_are_derived_from_persisted_elo_columns():
    frame = pd.DataFrame([{
        "team_elo_general_pre": 1600.0,
        "opponent_elo_general_pre": 1550.0,
        "team_elo_attack_pre": 0.30,
        "team_elo_defense_pre": 0.10,
        "opponent_elo_attack_pre": 0.20,
        "opponent_elo_defense_pre": -0.05,
        "player_offensive_elo_pre": 0.55,
        "player_defensive_elo_pre": 0.00,
        "team_lineup_attack_strength": 0.40,
        "team_lineup_defense_strength": 0.15,
    }])

    enriched = add_model_interaction_features(frame)

    assert enriched.iloc[0]["team_elo_general_diff"] == 50.0
    assert enriched.iloc[0]["team_attack_vs_opp_defense"] == pytest.approx(0.35)
    assert enriched.iloc[0]["player_attack_delta_vs_team"] == pytest.approx(0.25)
    assert enriched.iloc[0]["lineup_defense_vs_opp_attack"] == pytest.approx(-0.05)


def test_default_targets_cover_all_eleven_events_and_exclude_leaky_features():
    from football_analytics.ml_training import DEFAULT_TARGET_COLUMNS

    assert DEFAULT_TARGET_COLUMNS == [
        "offsides",
        "shots_total",
        "shots_on",
        "goals_total",
        "goals_assists",
        "goals_saves",
        "passes_total",
        "fouls_drawn",
        "fouls_committed",
        "cards_yellow",
        "cards_red",
    ]
    # games_minutes is post-match information; minutes enter via the exposure
    # offset, never as a feature.
    assert "games_minutes" not in DEFAULT_LIGHTGBM_FEATURES


def test_goals_saves_training_rows_are_restricted_to_goalkeepers():
    from football_analytics.ml_training import filter_rows_for_target

    frame = pd.DataFrame([
        {"player_id": 1, "is_goalkeeper": True, "position_group": "G", "goals_saves": 4},
        {"player_id": 2, "is_goalkeeper": False, "position_group": "F", "goals_saves": 0},
        {"player_id": 3, "is_goalkeeper": None, "position_group": "D", "goals_saves": 0},
    ])

    saves_rows = filter_rows_for_target(frame, "goals_saves")
    assert saves_rows["player_id"].tolist() == [1]

    # position_group fallback when is_goalkeeper is absent
    fallback_rows = filter_rows_for_target(
        frame.drop(columns=["is_goalkeeper"]), "goals_saves"
    )
    assert fallback_rows["player_id"].tolist() == [1]

    # non-gated targets keep every row
    all_rows = filter_rows_for_target(frame, "shots_total")
    assert all_rows["player_id"].tolist() == [1, 2, 3]


def test_build_training_feature_frame_keeps_only_completed_rows_with_minutes():
    frame = pd.DataFrame([
        {"fixture_id": 1, "is_completed_fixture": True, "games_minutes": 90.0},
        {"fixture_id": 2, "is_completed_fixture": False, "games_minutes": None},
        {"fixture_id": 3, "is_completed_fixture": True, "games_minutes": 0.0},
    ])

    training = build_training_feature_frame(frame, require_elo=False)

    assert training["fixture_id"].tolist() == [1]


def test_default_lightgbm_features_include_persisted_and_derived_elo_features():
    expected = {
        "player_offensive_rating_pre",
        "player_defensive_rating_pre",
        "team_lineup_attack_strength",
        "team_lineup_defense_strength",
        "team_elo_general_diff",
        "player_attack_vs_opp_defense",
        "lineup_attack_delta_vs_team",
    }

    assert expected.issubset(set(DEFAULT_LIGHTGBM_FEATURES))

    # M5 cleanup: entries that fct_football__player_event_features can never
    # produce were removed (they were silently dropped by
    # _select_available_features anyway — leftovers from the deleted
    # fct_football__player_shot_features mart).
    dead_features = {
        "is_starter",
        "was_substitute",
        "dribbles_attempts_l5_p90",
        "tackles_interceptions_l5_p90",
        "game_importance_l5",
        "opponent_strength_adjustment",
        "defensive_containment_rating",
        "opponent_defensive_elo_l10",
    }
    assert not dead_features & set(DEFAULT_LIGHTGBM_FEATURES)


def test_log_model_kwargs_include_signature_and_support_mlflow_versions():
    example = pd.DataFrame({"feature": [1.0]})
    signature = object()

    def mlflow3_log_model(model, name, registered_model_name=None, input_example=None, signature=None):
        return None

    def mlflow2_log_model(model, artifact_path, registered_model_name=None, input_example=None, signature=None):
        return None

    mlflow3_kwargs = _log_model_kwargs(
        mlflow3_log_model,
        model_name="shots_total_lightgbm_booster",
        registered_model_name="catalog.schema.model",
        input_example=example,
        signature=signature,
    )
    mlflow2_kwargs = _log_model_kwargs(
        mlflow2_log_model,
        model_name="shots_total_lightgbm_booster",
        registered_model_name="catalog.schema.model",
        input_example=example,
        signature=signature,
    )

    assert mlflow3_kwargs["name"] == "shots_total_lightgbm_booster"
    assert mlflow2_kwargs["artifact_path"] == "shots_total_lightgbm_booster"
    assert mlflow3_kwargs["input_example"].equals(example)
    assert mlflow3_kwargs["signature"] is signature
    assert mlflow3_kwargs["registered_model_name"] == "catalog.schema.model"


def test_assemble_hierarchical_feature_frame_overwrites_stale_elo_columns():
    features = pd.DataFrame([{
        "fixture_id": 1,
        "team_id": 1,
        "player_id": 10,
        "team_elo_general_pre": -999.0,
        "player_offensive_modifier_pre": -999.0,
        "shots_total": 1,
    }])
    team_history = pd.DataFrame([{
        "fixture_id": 1,
        "team_id": 1,
        "team_elo_general_pre": 1500.0,
        "opponent_elo_general_pre": 1490.0,
        "team_elo_attack_pre": 0.1,
        "team_elo_defense_pre": 0.2,
        "opponent_elo_attack_pre": -0.1,
        "opponent_elo_defense_pre": 0.0,
        "expected_goals_for_pre": 1.3,
        "expected_goals_against_pre": 1.1,
    }])
    player_history = pd.DataFrame([{
        "fixture_id": 1,
        "team_id": 1,
        "player_id": 10,
        "player_offensive_modifier_pre": 0.3,
        "player_defensive_modifier_pre": -0.2,
        "player_offensive_rating_pre": 0.4,
        "player_defensive_rating_pre": 0.0,
        "missed_fixture_count_pre": 0,
    }])

    enriched = assemble_hierarchical_feature_frame(features, team_history, player_history)

    assert enriched.iloc[0]["team_elo_general_pre"] == 1500.0
    assert enriched.iloc[0]["player_offensive_modifier_pre"] == 0.3


def test_season_split_trains_on_strictly_earlier_seasons_only():
    from football_analytics.ml_training import season_train_validation_split

    frame = pd.DataFrame({
        "fixture_id": [1, 2, 3, 4, 5],
        "league_season": [2022, 2023, 2024, 2024, 2025],
    })

    train, validation = season_train_validation_split(frame, validation_seasons=[2024])

    assert train["fixture_id"].tolist() == [1, 2]
    assert validation["fixture_id"].tolist() == [3, 4]
    # 2025 rows (after the validation season) appear on neither side
    assert 5 not in train["fixture_id"].tolist() + validation["fixture_id"].tolist()

    with pytest.raises(ValueError, match="No training rows"):
        season_train_validation_split(frame, validation_seasons=[2022])
    with pytest.raises(ValueError, match="No rows found"):
        season_train_validation_split(frame, validation_seasons=[2030])


def test_temporal_train_validation_split_keeps_future_rows_out_of_train():
    frame = pd.DataFrame({
        "fixture_id": [3, 1, 2, 4, 5],
        "fixture_date_utc": pd.to_datetime([
            "2023-01-03",
            "2023-01-01",
            "2023-01-02",
            "2023-01-04",
            "2023-01-05",
        ], utc=True),
    })

    train, validation = temporal_train_validation_split(frame, validation_fraction=0.4)

    assert train["fixture_id"].tolist() == [1, 2, 3]
    assert validation["fixture_id"].tolist() == [4, 5]
    assert train["fixture_date_utc"].max() < validation["fixture_date_utc"].min()
