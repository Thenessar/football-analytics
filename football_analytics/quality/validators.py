import re
from typing import Mapping

EXPECTED_WORLD_CUP_LEAGUE_ID = 1
EXPECTED_WORLD_CUP_SEASON = 2026
COMPLETED_STATUSES = {"FT", "AET", "PEN"}
SENIOR_MENS_INTERNATIONAL_LEAGUE_IDS = {
    1,   # FIFA World Cup
    4,   # UEFA Euro Championship
    5,   # UEFA Nations League
    6,   # Africa Cup of Nations
    9,   # Copa America
    10,  # senior international friendlies
}
SENIOR_MENS_INTERNATIONAL_NAME_PATTERNS = (
    "world cup",
    "euro championship",
    "euro qualification",
    "copa america",
    "africa cup of nations",
    "afcon",
    "asian cup",
    "gold cup",
    "nations league",
    "world cup qualification",
    "friendlies",
)
NON_SENIOR_MENS_TOKENS = (
    "women",
    "woman",
    "u17",
    "u18",
    "u19",
    "u20",
    "u21",
    "u22",
    "u23",
    "youth",
    "olympic",
    "club",
    "clubs",
)


class ValidationError(ValueError):
    """Raised when payloads are not safe for Silver writes."""


def _normalize_competition_label(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split())


def is_senior_mens_international_fixture(fixture: Mapping) -> bool:
    """
    Returns True for senior men's national-team competitions.

    API-Football league metadata can vary by season and competition family, so
    this uses a conservative allowlist plus defensive name-token exclusions for
    club, women's, youth, and Olympic events.
    """
    league = fixture.get("league") or {}
    league_id = league.get("id")
    league_name = _normalize_competition_label(league.get("name"))
    league_country = _normalize_competition_label(league.get("country"))
    combined = f"{league_name} {league_country}".strip()

    if any(token in combined for token in NON_SENIOR_MENS_TOKENS):
        return False
    if league_id in SENIOR_MENS_INTERNATIONAL_LEAGUE_IDS:
        return True
    return any(pattern in combined for pattern in SENIOR_MENS_INTERNATIONAL_NAME_PATTERNS)


def validate_senior_mens_international_fixture(
    fixture: Mapping,
    *,
    require_completed: bool = True,
) -> None:
    fixture_meta = fixture.get("fixture") or {}
    status = fixture_meta.get("status") or {}
    league = fixture.get("league") or {}
    if not is_senior_mens_international_fixture(fixture):
        raise ValidationError(
            f"Fixture {fixture_meta.get('id')} is not a senior men's international fixture: "
            f"league.id={league.get('id')} league.name={league.get('name')}"
        )
    if require_completed and status.get("short") not in COMPLETED_STATUSES:
        raise ValidationError(
            f"Fixture {fixture_meta.get('id')} has unsupported status {status.get('short')}"
        )


def validate_world_cup_fixture(
    fixture: Mapping,
    *,
    league_id: int = EXPECTED_WORLD_CUP_LEAGUE_ID,
    season: int = EXPECTED_WORLD_CUP_SEASON,
    require_completed: bool = True,
) -> None:
    league = fixture.get("league") or {}
    fixture_meta = fixture.get("fixture") or {}
    status = fixture_meta.get("status") or {}
    if league.get("id") != league_id or league.get("season") != season:
        raise ValidationError(
            f"Fixture {fixture_meta.get('id')} is not World Cup {season}: "
            f"league.id={league.get('id')} league.season={league.get('season')}"
        )
    if require_completed and status.get("short") not in COMPLETED_STATUSES:
        raise ValidationError(
            f"Fixture {fixture_meta.get('id')} has unsupported status {status.get('short')}"
        )

