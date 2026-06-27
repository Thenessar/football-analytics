import pytest

from football_analytics.databricks_ingestion import natural_key_merge_plans
from football_analytics.quality.validators import (
    SENIOR_MENS_NATIONAL_LEAGUE_IDS,
    ValidationError,
    validate_senior_mens_international_fixture,
    validate_world_cup_fixture,
)


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


def test_senior_mens_international_validation_accepts_friendlies_and_continental_competitions():
    friendly = {
        "fixture": {"id": 2001, "status": {"short": "FT"}},
        "league": {"id": 10, "season": 2024, "name": "Friendlies"},
    }
    afcon_qualifier = {
        "fixture": {"id": 2002, "status": {"short": "FT"}},
        "league": {"id": 36, "season": 2025, "name": "Africa Cup of Nations - Qualification"},
    }

    validate_senior_mens_international_fixture(friendly)
    validate_senior_mens_international_fixture(afcon_qualifier)


def test_senior_mens_international_validation_rejects_club_women_youth_and_olympics():
    rejected = [
        {"id": 667, "name": "Friendlies Clubs"},
        {"id": 666, "name": "Friendlies Women"},
        {"id": 490, "name": "World Cup - U20"},
        {"id": 882, "name": "Olympics Women - Qualification Asia"},
        {"id": 13, "name": "CONMEBOL Libertadores"},
    ]

    for league in rejected:
        fixture = {
            "fixture": {"id": league["id"], "status": {"short": "FT"}},
            "league": {"id": league["id"], "season": 2024, "name": league["name"]},
        }
        with pytest.raises(ValidationError, match="not an allowed senior men's"):
            validate_senior_mens_international_fixture(fixture)


def test_senior_mens_league_allowlist_documents_scope():
    assert 10 in SENIOR_MENS_NATIONAL_LEAGUE_IDS
    assert 960 in SENIOR_MENS_NATIONAL_LEAGUE_IDS
    assert 667 not in SENIOR_MENS_NATIONAL_LEAGUE_IDS
    assert 666 not in SENIOR_MENS_NATIONAL_LEAGUE_IDS
    assert 490 not in SENIOR_MENS_NATIONAL_LEAGUE_IDS


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
