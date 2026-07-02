import re
from typing import Mapping

from football_analytics.league_scope import allowed_league_ids


COMPLETED_STATUSES = {"FT", "AET", "PEN"}
SENIOR_MENS_NATIONAL_LEAGUE_IDS = allowed_league_ids()
SENIOR_MENS_INTERNATIONAL_LEAGUE_IDS = SENIOR_MENS_NATIONAL_LEAGUE_IDS
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


def is_senior_mens_international_fixture(
    fixture: Mapping,
    *,
    allowed_league_ids: set[int] = SENIOR_MENS_NATIONAL_LEAGUE_IDS,
) -> bool:
    """
    Returns True for reviewed senior men's national-team competitions.

    League IDs are the canonical allowlist. Name-token exclusions are a
    defensive guard against accidental provider metadata drift into club,
    women's, youth, or Olympic competitions.
    """
    league = fixture.get("league") or {}
    league_id = league.get("id")
    league_name = _normalize_competition_label(league.get("name"))
    league_country = _normalize_competition_label(league.get("country"))
    combined = f"{league_name} {league_country}".strip()

    if any(token in combined for token in NON_SENIOR_MENS_TOKENS):
        return False
    return league_id in allowed_league_ids


def validate_senior_mens_international_fixture(
    fixture: Mapping,
    *,
    allowed_league_ids: set[int] = SENIOR_MENS_NATIONAL_LEAGUE_IDS,
    require_completed: bool = True,
) -> None:
    fixture_meta = fixture.get("fixture") or {}
    status = fixture_meta.get("status") or {}
    league = fixture.get("league") or {}
    if not is_senior_mens_international_fixture(
        fixture,
        allowed_league_ids=allowed_league_ids,
    ):
        raise ValidationError(
            f"Fixture {fixture_meta.get('id')} is not an allowed senior men's "
            f"national-team competition: league.id={league.get('id')} league.name={league.get('name')}"
        )
    if require_completed and status.get("short") not in COMPLETED_STATUSES:
        raise ValidationError(
            f"Fixture {fixture_meta.get('id')} has unsupported status {status.get('short')}"
        )
