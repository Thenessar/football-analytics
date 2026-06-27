from typing import Mapping

EXPECTED_WORLD_CUP_LEAGUE_ID = 1
EXPECTED_WORLD_CUP_SEASON = 2026
COMPLETED_STATUSES = {"FT", "AET", "PEN"}
SENIOR_MENS_NATIONAL_LEAGUE_IDS = {
    1,    # World Cup
    4,    # Euro Championship
    5,    # UEFA Nations League
    6,    # Africa Cup of Nations
    7,    # Asian Cup
    9,    # Copa America
    10,   # Friendlies
    21,   # Confederations Cup
    22,   # CONCACAF Gold Cup
    29,   # World Cup - Qualification Africa
    30,   # World Cup - Qualification Asia
    31,   # World Cup - Qualification CONCACAF
    32,   # World Cup - Qualification Europe
    33,   # World Cup - Qualification Oceania
    34,   # World Cup - Qualification South America
    35,   # Asian Cup - Qualification
    36,   # Africa Cup of Nations - Qualification
    37,   # World Cup - Qualification Intercontinental Play-offs
    536,  # CONCACAF Nations League
    806,  # OFC Nations Cup
    808,  # CONCACAF Nations League - Qualification
    858,  # CONCACAF Gold Cup - Qualification
    913,  # CONMEBOL - UEFA Finalissima
    960,  # Euro Championship - Qualification
}


class ValidationError(ValueError):
    """Raised when payloads are not safe for Silver writes."""


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


def validate_senior_mens_international_fixture(
    fixture: Mapping,
    *,
    allowed_league_ids: set[int] = SENIOR_MENS_NATIONAL_LEAGUE_IDS,
    require_completed: bool = True,
) -> None:
    league = fixture.get("league") or {}
    fixture_meta = fixture.get("fixture") or {}
    status = fixture_meta.get("status") or {}
    league_id = league.get("id")
    if league_id not in allowed_league_ids:
        raise ValidationError(
            f"Fixture {fixture_meta.get('id')} is not an allowed senior men's "
            f"national-team competition: league.id={league_id} league.name={league.get('name')}"
        )
    if require_completed and status.get("short") not in COMPLETED_STATUSES:
        raise ValidationError(
            f"Fixture {fixture_meta.get('id')} has unsupported status {status.get('short')}"
        )
