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
    _prepare_xy,
    count_threshold_probabilities,
    evaluate_count_predictions,
    poisson_log_loss,
    temporal_train_validation_split,
)
from scripts.train_poisson_lgbm import build_training_feature_frame


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


def test_training_script_enriches_feature_frame_with_elo_columns():
    fixtures = pd.DataFrame([
        _fixture(1, "2023-01-01T12:00:00Z", home_goals=2, away_goals=0),
        _fixture(2, "2023-01-08T12:00:00Z", home_goals=1, away_goals=1),
    ])
    rankings = pd.DataFrame([
        {"Team": "Team A", "Raiting": 1600.0},
        {"Team": "Team B", "Raiting": 1500.0},
    ])
    feature_frame = pd.DataFrame([
        {
            "fixture_id": 1,
            "fixture_date_utc": "2023-01-01T12:00:00Z",
            "team_id": 1,
            "team_name": "Team A",
            "player_id": 10,
            "player_name": "Explosive Winger",
            "games_minutes": 90,
            "shots_total": 4,
            "shots_on": 2,
            "dribbles_attempts": 3,
            "goals_assists": 0,
            "tackles_interceptions": 0,
            "fouls_committed": 0,
            "team_elo_general_pre": -999.0,
        },
        {
            "fixture_id": 1,
            "fixture_date_utc": "2023-01-01T12:00:00Z",
            "team_id": 2,
            "team_name": "Team B",
            "player_id": 20,
            "player_name": "Defender",
            "games_minutes": 90,
            "shots_total": 0,
            "shots_on": 0,
            "dribbles_attempts": 0,
            "goals_assists": 0,
            "tackles_interceptions": 2,
            "fouls_committed": 1,
            "team_elo_general_pre": -999.0,
        },
        {
            "fixture_id": 2,
            "fixture_date_utc": "2023-01-08T12:00:00Z",
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
            "fouls_committed": 0,
            "team_elo_general_pre": -999.0,
        },
        {
            "fixture_id": 2,
            "fixture_date_utc": "2023-01-08T12:00:00Z",
            "team_id": 2,
            "team_name": "Team B",
            "player_id": 20,
            "player_name": "Defender",
            "games_minutes": 90,
            "shots_total": 0,
            "shots_on": 0,
            "dribbles_attempts": 0,
            "goals_assists": 0,
            "tackles_interceptions": 1,
            "fouls_committed": 1,
            "team_elo_general_pre": -999.0,
        },
    ])

    enriched = build_training_feature_frame(
        feature_frame,
        fixtures_frame=fixtures,
        rankings_frame=rankings,
        decay_alpha=0.85,
    )
    first_team_a = enriched[
        (enriched["fixture_id"] == 1) & (enriched["team_id"] == 1)
    ].iloc[0]
    second_team_a = enriched[
        (enriched["fixture_id"] == 2) & (enriched["team_id"] == 1)
    ].iloc[0]

    assert first_team_a["team_elo_general_pre"] == 1550.0
    assert first_team_a["team_elo_general_pre"] != -999.0
    assert second_team_a["team_elo_attack_pre"] > first_team_a["team_elo_attack_pre"]
    assert "player_offensive_modifier_pre" in enriched.columns
    assert "player_offensive_rating_pre" in enriched.columns


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
