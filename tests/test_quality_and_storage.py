import pytest

from football_analytics.league_scope import SENIOR_MENS_INTERNATIONAL_LEAGUES, allowed_league_ids
from football_analytics.quality.validators import (
    SENIOR_MENS_NATIONAL_LEAGUE_IDS,
    ValidationError,
    validate_senior_mens_international_fixture,
)


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
    assert SENIOR_MENS_NATIONAL_LEAGUE_IDS == allowed_league_ids()
    assert 10 in SENIOR_MENS_NATIONAL_LEAGUE_IDS
    assert 960 in SENIOR_MENS_NATIONAL_LEAGUE_IDS
    assert 667 not in SENIOR_MENS_NATIONAL_LEAGUE_IDS
    assert 666 not in SENIOR_MENS_NATIONAL_LEAGUE_IDS
    assert 490 not in SENIOR_MENS_NATIONAL_LEAGUE_IDS


def test_league_scope_source_of_truth_has_unique_ids():
    league_ids = [league_id for league_id, _ in SENIOR_MENS_INTERNATIONAL_LEAGUES]

    assert league_ids == sorted(league_ids)
    assert len(league_ids) == len(set(league_ids))
