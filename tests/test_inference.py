import numpy as np
import pandas as pd
import pytest

from football_analytics.inference import (
    INFERENCE_MODE_EXPLORATORY,
    INFERENCE_MODE_LIVE,
    LINEUP_SOURCE_CONFIRMED,
    LINEUP_SOURCE_PROJECTED,
    LoadedEventModel,
    MissingLineupError,
    build_prediction_records,
    deterministic_prediction_set_id,
    fixture_has_confirmed_starting_xi,
    lineup_signature,
    resolve_lineup_source,
)


class StubBooster:
    """Predicts a constant raw (log per-90) score for every row."""

    def __init__(self, raw_score: float = 0.0):
        self.raw_score = raw_score

    def predict(self, X, raw_score=False):
        return np.full(len(X), self.raw_score, dtype=float)


def _fixture_rows(starters_per_team=11, teams=(1, 2)):
    rows = []
    player_id = 0
    for team_id in teams:
        for slot in range(starters_per_team):
            player_id += 1
            rows.append({
                "fixture_id": 100,
                "fixture_date_utc": pd.Timestamp("2026-07-10T18:00:00Z"),
                "team_id": team_id,
                "team_name": f"Team {team_id}",
                "opponent_team_id": next((other for other in teams if other != team_id), 99),
                "opponent_team_name": "Opponent",
                "home_away": "home" if team_id == teams[0] else "away",
                "player_id": player_id,
                "player_name": f"Player {player_id}",
                "position_group": "G" if slot == 0 else "M",
                "is_goalkeeper": slot == 0,
                "is_starting": True,
                "formation": "4-3-3",
                "expected_minutes": 90.0,
                "team_elo_general_pre": 1500.0,
            })
    return pd.DataFrame(rows)


def test_confirmed_starting_xi_requires_eleven_starters_on_both_teams():
    assert fixture_has_confirmed_starting_xi(_fixture_rows())
    assert not fixture_has_confirmed_starting_xi(_fixture_rows(starters_per_team=10))
    assert not fixture_has_confirmed_starting_xi(_fixture_rows(teams=(1,)))
    assert not fixture_has_confirmed_starting_xi(pd.DataFrame())


def test_live_mode_blocks_unconfirmed_lineups_and_exploratory_tags_projected():
    assert (
        resolve_lineup_source(INFERENCE_MODE_LIVE, lineups_confirmed=True)
        == LINEUP_SOURCE_CONFIRMED
    )
    assert (
        resolve_lineup_source(INFERENCE_MODE_EXPLORATORY, lineups_confirmed=False)
        == LINEUP_SOURCE_PROJECTED
    )
    with pytest.raises(MissingLineupError, match="live-mode inference is skipped"):
        resolve_lineup_source(INFERENCE_MODE_LIVE, lineups_confirmed=False)


def test_prediction_set_id_is_deterministic_and_lineup_sensitive():
    rows = _fixture_rows()
    digest = lineup_signature(rows)
    versions = {"shots_total": "3"}

    first = deterministic_prediction_set_id(100, lineup_digest=digest, model_versions=versions)
    second = deterministic_prediction_set_id(100, lineup_digest=digest, model_versions=versions)
    assert first == second

    changed_rows = rows.copy()
    changed_rows.loc[changed_rows.index[-1], "player_id"] = 999
    changed = deterministic_prediction_set_id(
        100, lineup_digest=lineup_signature(changed_rows), model_versions=versions
    )
    assert changed != first

    new_version = deterministic_prediction_set_id(
        100, lineup_digest=digest, model_versions={"shots_total": "4"}
    )
    assert new_version != first


def test_prediction_records_shape_probabilities_and_goalkeeper_gating():
    rows = _fixture_rows()
    models = [
        LoadedEventModel(
            target_event="shots_total",
            booster=StubBooster(raw_score=0.0),
            feature_columns=("team_elo_general_pre",),
            model_name="catalog.gold.player_event__shots_total",
            model_version="3",
        ),
        LoadedEventModel(
            target_event="goals_saves",
            booster=StubBooster(raw_score=np.log(3.0)),
            feature_columns=("team_elo_general_pre",),
            model_name="catalog.gold.player_event__goals_saves",
            model_version="2",
            goalkeeper_only=True,
        ),
    ]

    records = build_prediction_records(
        rows,
        models,
        prediction_set_id="set-1",
        prediction_run_id="run-1",
        lineup_source=LINEUP_SOURCE_CONFIRMED,
        feature_table_name="catalog.gold.fct_football__player_event_features",
        feature_table_version="42",
    )

    assert len(records) == len(rows) * len(models)
    assert set(records["target_event"]) == {"shots_total", "goals_saves"}
    assert records["is_active_prediction"].all()
    assert (records["lineup_source"] == LINEUP_SOURCE_CONFIRMED).all()

    shots = records[records["target_event"] == "shots_total"]
    # raw score 0 with 90 expected minutes -> mean exp(0) * 1.0 exposure
    assert shots["predicted_mean"].to_numpy() == pytest.approx(np.ones(len(shots)))
    assert (shots["predicted_p_ge_1"] >= shots["predicted_p_ge_2"]).all()
    assert (shots["predicted_p_ge_2"] >= shots["predicted_p_ge_3"]).all()

    saves = records[records["target_event"] == "goals_saves"]
    outfield = saves[~saves["position_group"].eq("G")]
    goalkeepers = saves[saves["position_group"].eq("G")]
    assert (outfield["predicted_mean"] == 0.0).all()
    assert (outfield["predicted_p_ge_1"] == 0.0).all()
    assert (goalkeepers["predicted_mean"] > 0.0).all()

    assert (records["model_version"].isin({"3", "2"})).all()
    assert (records["feature_table_version"] == "42").all()


def test_prediction_records_reject_mixed_fixtures():
    rows = _fixture_rows()
    rows.loc[rows.index[-1], "fixture_id"] = 200

    with pytest.raises(ValueError, match="exactly one fixture"):
        build_prediction_records(
            rows,
            [],
            prediction_set_id="set-1",
            prediction_run_id="run-1",
            lineup_source=LINEUP_SOURCE_CONFIRMED,
            feature_table_name="table",
        )
