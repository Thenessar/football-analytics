import csv
import json
from pathlib import Path
from typing import Iterable, Mapping


SENIOR_MENS_INTERNATIONAL_LEAGUES = (
    (1, "World Cup"),
    (4, "Euro Championship"),
    (5, "UEFA Nations League"),
    (6, "Africa Cup of Nations"),
    (7, "Asian Cup"),
    (9, "Copa America"),
    (10, "Friendlies"),
    (21, "Confederations Cup"),
    (22, "CONCACAF Gold Cup"),
    (29, "World Cup - Qualification Africa"),
    (30, "World Cup - Qualification Asia"),
    (31, "World Cup - Qualification CONCACAF"),
    (32, "World Cup - Qualification Europe"),
    (33, "World Cup - Qualification Oceania"),
    (34, "World Cup - Qualification South America"),
    (35, "Asian Cup - Qualification"),
    (36, "Africa Cup of Nations - Qualification"),
    (37, "World Cup - Qualification Intercontinental Play-offs"),
    (536, "CONCACAF Nations League"),
    (806, "OFC Nations Cup"),
    (808, "CONCACAF Nations League - Qualification"),
    (858, "CONCACAF Gold Cup - Qualification"),
    (913, "CONMEBOL - UEFA Finalissima"),
    (960, "Euro Championship - Qualification"),
)

SEED_COLUMNS = ("league_id", "league_name")


def allowed_league_ids(leagues: Iterable[tuple[int, str]] = SENIOR_MENS_INTERNATIONAL_LEAGUES) -> set[int]:
    return {int(league_id) for league_id, _ in leagues}


def distinct_leagues_from_api_response(payload: Mapping) -> list[tuple[int, str]]:
    leagues = {
        (int(entry["league"]["id"]), str(entry["league"]["name"]))
        for entry in payload.get("response", [])
        if entry.get("league") and entry["league"].get("id") is not None
    }
    return sorted(leagues, key=lambda item: item[0])


def load_distinct_leagues_from_response_file(path: str | Path) -> list[tuple[int, str]]:
    with Path(path).open(encoding="utf-8") as handle:
        return distinct_leagues_from_api_response(json.load(handle))


def write_dbt_seed(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SEED_COLUMNS)
        writer.writerows(SENIOR_MENS_INTERNATIONAL_LEAGUES)
