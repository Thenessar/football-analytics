import os
import pandas as pd
import pytest
from football_analytics.pipeline import (
    FootballDataPipeline,
    ConfirmedLineupDataError,
    FixtureResolutionError,
)

@pytest.fixture
def offline_pipeline():
    return FootballDataPipeline(offline=True, cache_file="world_cup_2026_completed_data.json")

def test_resolve_team_id(offline_pipeline):
    """Checks that offline pipeline correctly falls back or finds team ID."""
    ger_id = offline_pipeline.find_team_id("Germany")
    ecu_id = offline_pipeline.find_team_id("Ecuador")
    assert ger_id == 25
    assert ecu_id == 2382

def test_load_historical_team_stats(offline_pipeline):
    """Verifies that stats are loaded, minimized, and nulls are cleaned to 0."""
    df_ger = offline_pipeline.load_historical_team_stats("Germany", limit=5)
    
    # Verify Data Minimization payload features
    assert not df_ger.empty
    expected_cols = [
        "fixture_id", "fixture_date", "match_home", "match_away",
        "fulltime_score_home", "fulltime_score_away",
        "player_id", "player_name", "games_minutes", "games_position",
        "shots_total", "shots_on", "goals_total"
    ]
    for col in expected_cols:
        assert col in df_ger.columns
        
    # Verify Data Integrity Guard: No nulls in numeric shot/goal properties
    assert not df_ger["shots_total"].isnull().any()
    assert not df_ger["shots_on"].isnull().any()
    assert not df_ger["goals_total"].isnull().any()
    
    # Ensure they are integer types
    assert df_ger["shots_total"].dtype in ["int64", "int32"]
    assert df_ger["shots_on"].dtype in ["int64", "int32"]
    assert df_ger["goals_total"].dtype in ["int64", "int32"]


def test_historical_stats_exclude_target_fixture(offline_pipeline):
    df_ger = offline_pipeline.load_historical_team_stats(
        "Germany",
        limit=5,
        exclude_fixture_id=1489393,
    )

    assert 1489393 not in df_ger["fixture_id"].unique()


def test_offline_lineups_do_not_fabricate_a_historical_starting_xi(offline_pipeline):
    """Missing fixture lineups must not silently become the first 11 cached players."""
    assert offline_pipeline.fetch_confirmed_lineups(1489437) is None


def test_live_lineup_fetch_fails_closed_when_api_has_no_sheet(monkeypatch, tmp_path):
    class EmptyLineupResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": []}

    monkeypatch.setattr(
        "football_analytics.pipeline.requests.get",
        lambda *args, **kwargs: EmptyLineupResponse(),
    )
    pipeline = FootballDataPipeline(
        offline=False,
        cache_file=str(tmp_path / "cache.json"),
    )

    with pytest.raises(ConfirmedLineupDataError, match="unavailable"):
        pipeline.fetch_confirmed_lineups(1489437)


def test_resolve_fixture_by_utc_date_with_team_aliases(monkeypatch, tmp_path):
    class FixtureResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "errors": [],
                "response": [{
                    "fixture": {
                        "id": 1489409,
                        "date": "2026-06-25T20:00:00+00:00",
                    },
                    "teams": {
                        "home": {"name": "Curaçao"},
                        "away": {"name": "Ivory Coast"},
                    },
                }],
            }

    monkeypatch.setattr(
        "football_analytics.pipeline.requests.get",
        lambda *args, **kwargs: FixtureResponse(),
    )
    pipeline = FootballDataPipeline(
        offline=False,
        cache_file=str(tmp_path / "cache.json"),
    )

    resolved = pipeline.resolve_fixture_by_date(
        "Curacao",
        "Cote d'lvoire",
        "2026-06-25",
    )

    assert resolved["fixture_id"] == 1489409
    assert resolved["home_team"] == "Curaçao"
    assert resolved["away_team"] == "Ivory Coast"


def test_resolve_fixture_by_date_detects_reversed_order(monkeypatch, tmp_path):
    class FixtureResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "errors": [],
                "response": [{
                    "fixture": {
                        "id": 1489410,
                        "date": "2026-06-25T20:00:00+00:00",
                    },
                    "teams": {
                        "home": {"name": "Ecuador"},
                        "away": {"name": "Germany"},
                    },
                }],
            }

    monkeypatch.setattr(
        "football_analytics.pipeline.requests.get",
        lambda *args, **kwargs: FixtureResponse(),
    )
    pipeline = FootballDataPipeline(
        offline=False,
        cache_file=str(tmp_path / "cache.json"),
    )

    with pytest.raises(FixtureResolutionError, match="reversed home/away"):
        pipeline.resolve_fixture_by_date(
            "Germany",
            "Ecuador",
            "2026-06-25",
        )


def test_resolve_fixture_rejects_invalid_utc_date(offline_pipeline):
    with pytest.raises(FixtureResolutionError, match="YYYY-MM-DD"):
        offline_pipeline.resolve_fixture_by_date(
            "Germany",
            "Ecuador",
            "25-06-2026",
        )
