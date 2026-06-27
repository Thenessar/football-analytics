import pytest

from football_analytics.databricks_ingestion import natural_key_merge_plans
from football_analytics.quality.validators import ValidationError, validate_world_cup_fixture


def test_non_world_cup_fixture_1036663_is_rejected():
    fixture = {
        "fixture": {"id": 1036663, "status": {"short": "FT"}},
        "league": {"id": 667, "season": 2023, "name": "Club Friendlies"},
    }

    with pytest.raises(ValidationError, match="not World Cup 2026"):
        validate_world_cup_fixture(fixture)


def test_world_cup_fixture_validation_accepts_completed_fixture():
    fixture = {
        "fixture": {"id": 1489437, "status": {"short": "FT"}},
        "league": {"id": 1, "season": 2026, "name": "World Cup"},
    }

    validate_world_cup_fixture(fixture)


def test_idempotent_merge_plans_use_required_natural_keys():
    plans = natural_key_merge_plans()

    assert plans["fixtures"].keys == ("fixture_id",)
    assert plans["player_stats"].keys == ("fixture_id", "team_id", "player_id")
    assert plans["lineups"].keys == ("fixture_id", "team_id", "player_id")
    assert plans["player_stats"].predicate == (
        "target.fixture_id = source.fixture_id AND "
        "target.team_id = source.team_id AND "
        "target.player_id = source.player_id"
    )

