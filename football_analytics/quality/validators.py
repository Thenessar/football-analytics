from typing import Mapping

EXPECTED_WORLD_CUP_LEAGUE_ID = 1
EXPECTED_WORLD_CUP_SEASON = 2026
COMPLETED_STATUSES = {"FT", "AET", "PEN"}


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

