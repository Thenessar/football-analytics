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


class _FakeWriter:
    def __init__(self, log):
        self._log = log

    def format(self, _fmt):
        return self

    def mode(self, mode):
        self._log["mode"] = mode
        return self

    def option(self, _key, _value):
        return self

    def saveAsTable(self, table):
        self._log["table"] = table


class _FakeColumn:
    def cast(self, _data_type):
        return self


class _FakeFrame:
    def __init__(self, log, columns):
        self.write = _FakeWriter(log)
        self.columns = list(columns)
        self.casts = log.setdefault("casts", [])

    def __getitem__(self, _name):
        return _FakeColumn()

    def withColumn(self, name, _column):
        self.casts.append(name)
        return self


class _FakeSpark:
    def __init__(self):
        self.sql_statements = []
        self.write_log = {}

    def sql(self, statement):
        self.sql_statements.append(" ".join(statement.split()))

    def createDataFrame(self, records):
        return _FakeFrame(self.write_log, records.columns)


def test_prediction_write_appends_and_flips_older_active_sets():
    from football_analytics.inference import write_player_event_predictions

    records = pd.DataFrame([
        {
            "prediction_set_id": "set-1",
            "fixture_id": 100,
            "model_name": "catalog.gold.player_event__shots_total",
            "model_version": "3",
            "is_active_prediction": True,
        },
        {
            "prediction_set_id": "set-1",
            "fixture_id": 100,
            "model_name": "catalog.gold.player_event__goals_saves",
            "model_version": "2",
            "is_active_prediction": True,
        },
    ])
    spark = _FakeSpark()

    summary = write_player_event_predictions(
        spark, records, prediction_table="catalog.gold.pred_football__player_event_predictions"
    )

    assert summary == {"written": 2, "deactivated_sets": 2}
    assert spark.write_log["mode"] == "append"
    assert spark.write_log["table"] == "catalog.gold.pred_football__player_event_predictions"
    # Key columns are cast to the DDL types so Delta never merges mismatched
    # field types (Python ints otherwise arrive as LONG vs the table's INT).
    assert "fixture_id" in spark.write_log["casts"]
    assert "is_active_prediction" in spark.write_log["casts"]

    create = spark.sql_statements[0]
    assert "CREATE TABLE IF NOT EXISTS" in create
    for column in ("prediction_set_id", "target_event", "predicted_p_ge_3", "is_active_prediction", "lineup_source"):
        assert column in create

    delete = spark.sql_statements[1]
    assert delete.startswith("DELETE FROM")
    assert "prediction_set_id IN ('set-1')" in delete

    updates = [s for s in spark.sql_statements if s.startswith("UPDATE")]
    assert len(updates) == 2
    for update in updates:
        assert "SET is_active_prediction = false" in update
        assert "prediction_set_id != 'set-1'" in update
        assert "fixture_id = 100" in update
    assert any("model_version = '3'" in update for update in updates)
    assert any("model_version = '2'" in update for update in updates)


def test_prediction_write_with_no_records_is_a_noop():
    from football_analytics.inference import write_player_event_predictions

    spark = _FakeSpark()
    summary = write_player_event_predictions(
        spark, pd.DataFrame(), prediction_table="catalog.gold.predictions"
    )

    assert summary == {"written": 0, "deactivated_sets": 0}
    assert spark.sql_statements == []


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
