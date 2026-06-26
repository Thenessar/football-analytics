import os
import json
import time
import datetime
import unicodedata
import requests
import pandas as pd
from typing import Dict, List, Optional
from football_analytics.config import (
    BASE_URL,
    CACHE_FILE,
    HEADERS,
    HISTORICAL_ANCHOR_DATE,
)


class ConfirmedLineupDataError(RuntimeError):
    """Raised when a live run cannot obtain a valid official starting XI payload."""


class FixtureResolutionError(RuntimeError):
    """Raised when teams and a UTC date cannot resolve to one fixture."""


TEAM_NAME_ALIASES = {
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "cote d lvoire": "ivory coast",
    "ivory coast": "ivory coast",
}


def _normalize_text(value: str) -> str:
    """Normalizes external labels for resilient team-name comparisons."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = "".join(char if char.isalnum() else " " for char in text)
    normalized = " ".join(text.casefold().split())
    return TEAM_NAME_ALIASES.get(normalized, normalized)


class FootballDataPipeline:
    def __init__(self, api_key: Optional[str] = None, cache_file: str = CACHE_FILE, offline: bool = False):
        self.api_key = api_key
        self.base_url = BASE_URL
        self.headers = HEADERS.copy()
        if api_key:
            self.headers["x-rapidapi-key"] = api_key
            self.headers["x-apisports-key"] = api_key
        self.cache_file = cache_file
        self.offline = offline
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Loads historical fixtures and player statistics cache."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}. Starting fresh.")
        return {}

    def _save_cache(self):
        """Saves current cache back to local file."""
        if self.offline:
            return
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving cache: {e}")

    @staticmethod
    def iter_date_chunks(
        start_date: str = HISTORICAL_ANCHOR_DATE,
        end_date: Optional[str] = None,
        chunk_days: int = 7,
    ) -> List[Dict[str, str]]:
        """Builds inclusive date chunks for historical backfill jobs."""
        if chunk_days < 1:
            raise ValueError("chunk_days must be >= 1")
        start = datetime.date.fromisoformat(start_date)
        end = (
            datetime.date.fromisoformat(end_date)
            if end_date
            else datetime.date.today()
        )
        if end < start:
            raise ValueError("end_date must be on or after start_date")

        chunks = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + datetime.timedelta(days=chunk_days - 1), end)
            chunks.append({
                "date_from": cursor.isoformat(),
                "date_to": chunk_end.isoformat(),
            })
            cursor = chunk_end + datetime.timedelta(days=1)
        return chunks

    def fetch_international_fixtures_by_date(
        self,
        match_date: str,
        completed_only: bool = True,
    ) -> List[Dict]:
        """Fetches all international fixtures for a UTC date."""
        if self.offline:
            fixtures = []
            for fixture_id, match_data in self.cache.items():
                match_info = match_data.get("match_info", {})
                if str(match_info.get("date") or "")[:10] != match_date:
                    continue
                fixtures.append({
                    "fixture": {
                        "id": int(fixture_id),
                        "date": match_info.get("date"),
                        "status": {"short": "FT"},
                    },
                    "league": match_info.get("league", {}),
                    "teams": {
                        "home": {"name": match_info.get("home_team")},
                        "away": {"name": match_info.get("away_team")},
                    },
                    "goals": match_info.get("score", {}),
                    "score": {"fulltime": match_info.get("score", {})},
                })
            return fixtures

        url = f"{self.base_url}/fixtures"
        params = {"date": match_date, "timezone": "UTC"}
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("errors"):
            raise Exception(f"API Error: {data['errors']}")

        fixtures = data.get("response", [])
        if not completed_only:
            return fixtures
        return [
            fixture
            for fixture in fixtures
            if ((fixture.get("fixture") or {}).get("status") or {}).get("short")
            in {"FT", "AET", "PEN"}
        ]

    def backfill_international_fixtures(
        self,
        start_date: str = HISTORICAL_ANCHOR_DATE,
        end_date: Optional[str] = None,
        chunk_days: int = 7,
        sleep_seconds: float = 1.0,
    ) -> List[Dict]:
        """
        Incrementally fetches completed international fixtures by date.

        Cached fixture IDs are skipped, making reruns idempotent for local
        development and mirroring the Databricks bronze checkpoint behavior.
        """
        backfilled = []
        cache_dirty = False
        for chunk in self.iter_date_chunks(start_date, end_date, chunk_days):
            for match_day in self.iter_date_chunks(
                chunk["date_from"],
                chunk["date_to"],
                1,
            ):
                fixtures = self.fetch_international_fixtures_by_date(
                    match_day["date_from"],
                    completed_only=True,
                )
                for fixture in fixtures:
                    fixture_id = (fixture.get("fixture") or {}).get("id")
                    if not fixture_id:
                        continue
                    fid_str = str(fixture_id)
                    if fid_str in self.cache and "player_statistics" in self.cache[fid_str]:
                        continue
                    player_stats = self.get_player_statistics(int(fixture_id))
                    self.cache[fid_str] = {
                        "match_info": {
                            "date": (fixture.get("fixture") or {}).get("date"),
                            "home_team": ((fixture.get("teams") or {}).get("home") or {}).get("name"),
                            "away_team": ((fixture.get("teams") or {}).get("away") or {}).get("name"),
                            "score": fixture.get("goals") or {},
                            "league": fixture.get("league") or {},
                        },
                        "fixture": fixture.get("fixture"),
                        "league": fixture.get("league"),
                        "teams": fixture.get("teams"),
                        "goals": fixture.get("goals"),
                        "score": fixture.get("score"),
                        "player_statistics": player_stats,
                    }
                    backfilled.append(self.cache[fid_str])
                    cache_dirty = True
                    if sleep_seconds:
                        time.sleep(sleep_seconds)
        if cache_dirty:
            self._save_cache()
        return backfilled

    def find_team_id(self, team_name: str) -> Optional[int]:
        """Looks up a team ID by name from cache or API search."""
        normalized_team_name = _normalize_text(team_name)

        # 1. Check local cache
        for match in self.cache.values():
            for team_stat in match.get("player_statistics", []):
                t_name = team_stat["team"]["name"]
                t_id = team_stat["team"]["id"]
                if _normalize_text(t_name) == normalized_team_name:
                    return t_id

        # 2. Hardcoded fallback values for target teams
        if normalized_team_name == "germany":
            return 25
        elif normalized_team_name == "ecuador":
            return 2382

        if self.offline:
            print(f"Offline Mode: Could not resolve Team ID for '{team_name}' (no cache hit). Using mock ID 999.")
            return 999

        # 3. Query API search
        url = f"{self.base_url}/teams"
        search_name = (
            "Ivory Coast"
            if normalized_team_name == "ivory coast"
            else team_name
        )
        params = {"search": search_name}
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            for item in data.get("response", []):
                team_info = item.get("team", {})
                if _normalize_text(team_info.get("name")) == normalized_team_name:
                    return team_info.get("id")
        except Exception as e:
            print(f"API request failed when searching team: {e}")
        return None

    def resolve_fixture_by_date(
        self,
        home_team: str,
        away_team: str,
        fixture_date: str,
    ) -> Dict:
        """Resolves exactly one fixture by home team, away team, and UTC date."""
        try:
            parsed_date = datetime.date.fromisoformat(fixture_date)
        except (TypeError, ValueError) as error:
            raise FixtureResolutionError(
                f"Invalid UTC fixture date '{fixture_date}'. Use YYYY-MM-DD."
            ) from error

        normalized_home = _normalize_text(home_team)
        normalized_away = _normalize_text(away_team)
        if not normalized_home or not normalized_away:
            raise FixtureResolutionError("Both home and away team names are required.")
        if normalized_home == normalized_away:
            raise FixtureResolutionError("Home and away teams must be different.")

        fixtures = []
        if self.offline:
            for fixture_id, match_data in self.cache.items():
                match_info = match_data.get("match_info", {})
                date_value = str(match_info.get("date") or "")
                if date_value[:10] != parsed_date.isoformat():
                    continue
                fixtures.append({
                    "fixture": {
                        "id": int(fixture_id),
                        "date": date_value,
                    },
                    "teams": {
                        "home": {"name": match_info.get("home_team", "")},
                        "away": {"name": match_info.get("away_team", "")},
                    },
                })
        else:
            url = f"{self.base_url}/fixtures"
            params = {
                "date": parsed_date.isoformat(),
                "timezone": "UTC",
            }
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as error:
                raise FixtureResolutionError(
                    f"Fixture lookup failed for {parsed_date.isoformat()} UTC: {error}"
                ) from error
            if data.get("errors"):
                raise FixtureResolutionError(
                    f"Fixture lookup API error for {parsed_date.isoformat()} UTC: "
                    f"{data['errors']}"
                )
            fixtures = data.get("response", [])

        exact_matches = []
        reversed_matches = []
        for item in fixtures:
            teams = item.get("teams", {})
            api_home = teams.get("home", {}).get("name", "")
            api_away = teams.get("away", {}).get("name", "")
            home_key = _normalize_text(api_home)
            away_key = _normalize_text(api_away)
            if home_key == normalized_home and away_key == normalized_away:
                exact_matches.append(item)
            elif home_key == normalized_away and away_key == normalized_home:
                reversed_matches.append(item)

        if not exact_matches:
            if reversed_matches:
                reverse = reversed_matches[0]
                teams = reverse.get("teams", {})
                raise FixtureResolutionError(
                    "Fixture found with reversed home/away order: "
                    f"{teams.get('home', {}).get('name')} vs "
                    f"{teams.get('away', {}).get('name')} on "
                    f"{parsed_date.isoformat()} UTC."
                )
            raise FixtureResolutionError(
                f"No fixture found for {home_team} vs {away_team} on "
                f"{parsed_date.isoformat()} UTC."
            )

        if len(exact_matches) > 1:
            fixture_ids = [
                str(item.get("fixture", {}).get("id"))
                for item in exact_matches
            ]
            raise FixtureResolutionError(
                f"Multiple fixtures matched {home_team} vs {away_team} on "
                f"{parsed_date.isoformat()} UTC: {', '.join(fixture_ids)}. "
                "Use --fixture-id."
            )

        match = exact_matches[0]
        fixture = match.get("fixture", {})
        teams = match.get("teams", {})
        fixture_id = fixture.get("id")
        if not fixture_id:
            raise FixtureResolutionError("Matched fixture has no API fixture ID.")

        return {
            "fixture_id": int(fixture_id),
            "fixture_date": fixture.get("date") or parsed_date.isoformat(),
            "home_team": teams.get("home", {}).get("name") or home_team,
            "away_team": teams.get("away", {}).get("name") or away_team,
        }

    def fetch_team_fixtures(self, team_id: int, limit: int = 5) -> List[Dict]:
        """Fetches completed fixtures for a team (league=1, season=2026)."""
        if self.offline:
            completed = []
            for fid, match_data in self.cache.items():
                player_stats = match_data.get("player_statistics", [])
                if len(player_stats) >= 2:
                    team1_id = player_stats[0]["team"]["id"]
                    team2_id = player_stats[1]["team"]["id"]
                    if team1_id == team_id or team2_id == team_id:
                        # Reconstruct layout that matches live format
                        completed.append({
                            "fixture": {
                                "id": int(fid),
                                "date": match_data.get("match_info", {}).get("date"),
                                "status": {"short": "FT"}
                            },
                            "teams": {
                                "home": {"id": team1_id, "name": player_stats[0]["team"]["name"]},
                                "away": {"id": team2_id, "name": player_stats[1]["team"]["name"]}
                            },
                            "goals": {
                                "home": match_data.get("match_info", {}).get("score", {}).get("home"),
                                "away": match_data.get("match_info", {}).get("score", {}).get("away")
                            }
                        })
            # Sort by date descending
            completed.sort(key=lambda x: x["fixture"]["date"] or "", reverse=True)
            return completed[:limit]

        # Live Mode
        url = f"{self.base_url}/fixtures"
        params = {"team": team_id, "season": 2026, "league": 1}
        print(f"Fetching fixtures for team {team_id}...")
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("errors"):
            raise Exception(f"API Error: {data['errors']}")
            
        fixtures = data.get("response", [])
        completed = [f for f in fixtures if f["fixture"]["status"]["short"] == "FT"]
        completed.sort(key=lambda x: x["fixture"]["date"], reverse=True)
        return completed[:limit]

    def get_player_statistics(self, fixture_id: int) -> List[Dict]:
        """Fetches player statistics for a fixture with local caching."""
        fid_str = str(fixture_id)
        if fid_str in self.cache and "player_statistics" in self.cache[fid_str]:
            return self.cache[fid_str].get("player_statistics", [])

        if self.offline:
            return []

        # Live Mode
        url = f"{self.base_url}/fixtures/players"
        params = {"fixture": fixture_id}
        print(f"Fetching player statistics for fixture {fixture_id} from API...")
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("errors"):
            print(f"API Error for fixture {fixture_id}: {data['errors']}")
            return []
            
        return data.get("response", [])

    def get_fixture_details(self, fixture_id: int) -> Dict:
        """Fetches general fixture metadata."""
        fid_str = str(fixture_id)
        if fid_str in self.cache and (
            "match_info" in self.cache[fid_str]
            or "player_statistics" in self.cache[fid_str]
        ):
            match_data = self.cache[fid_str]
            player_stats = match_data.get("player_statistics", [])
            match_info = match_data.get("match_info", {})
            home_name = match_info.get("home_team", "")
            away_name = match_info.get("away_team", "")
            team_ids = {
                _normalize_text(team.get("team", {}).get("name")): team.get("team", {}).get("id")
                for team in player_stats
            }
            
            return {
                "fixture": {
                    "id": fixture_id,
                    "date": match_info.get("date"),
                    "status": {"short": "FT"}
                },
                "teams": {
                    "home": {"id": team_ids.get(_normalize_text(home_name)), "name": home_name},
                    "away": {"id": team_ids.get(_normalize_text(away_name)), "name": away_name}
                },
                "score": {
                    "fulltime": {
                        "home": match_info.get("score", {}).get("home"),
                        "away": match_info.get("score", {}).get("away")
                    }
                }
            }

        if self.offline:
            return {}

        # Live Mode
        url = f"{self.base_url}/fixtures"
        params = {"id": fixture_id}
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        res = data.get("response", [])
        return res[0] if res else {}

    def fetch_confirmed_lineups(self, fixture_id: int) -> Optional[List[Dict]]:
        """
        Queries the fixture/lineups endpoint for the official starting XI.

        Offline runs may return no cached lineup so the explicit offline fixture
        can be used by the orchestrator. Live runs fail closed: API errors,
        malformed responses, and absent lineup sheets raise a fatal data error.
        """
        if self.offline:
            fid_str = str(fixture_id)
            cached_lineups = self.cache.get(fid_str, {}).get("lineups")
            return cached_lineups or None

        url = f"{self.base_url}/fixtures/lineups"
        params = {"fixture": fixture_id}
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as error:
            raise ConfirmedLineupDataError(
                f"Fatal live lineup fetch failure for fixture {fixture_id}: {error}"
            ) from error

        if data.get("errors"):
            raise ConfirmedLineupDataError(
                f"Fatal live lineup API error for fixture {fixture_id}: {data['errors']}"
            )

        lineups = data.get("response")
        if not isinstance(lineups, list) or not lineups:
            raise ConfirmedLineupDataError(
                f"Official starting XI sheets are unavailable for fixture {fixture_id}; "
                "report generation halted."
            )

        self.cache.setdefault(str(fixture_id), {})["lineups"] = lineups
        self._save_cache()
        return lineups

    def load_historical_team_stats(
        self,
        team_name: str,
        limit: int = 5,
        exclude_fixture_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Ingests, aggregates, minimizes, and cleanses the raw statistics 
        for a team's squad members across the trailing N completed games.
        """
        team_id = self.find_team_id(team_name)
        if not team_id:
            raise ValueError(f"Could not find Team ID for '{team_name}'")

        fixture_limit = limit + 1 if exclude_fixture_id is not None else limit
        completed_fixtures = self.fetch_team_fixtures(
            team_id,
            limit=fixture_limit,
        )
        if exclude_fixture_id is not None:
            completed_fixtures = [
                fixture
                for fixture in completed_fixtures
                if fixture.get("fixture", {}).get("id") != exclude_fixture_id
            ]
        completed_fixtures = completed_fixtures[:limit]
        
        records = []
        cache_dirty = False
        
        for f in completed_fixtures:
            fid = f["fixture"]["id"]
            fid_str = str(fid)
            
            if (
                fid_str not in self.cache
                or "player_statistics" not in self.cache[fid_str]
            ):
                if self.offline:
                    continue
                player_stats = self.get_player_statistics(fid)
                meta = self.get_fixture_details(fid)
                time.sleep(1.0)
                self.cache[fid_str] = {
                    "player_statistics": player_stats
                }
                cache_dirty = True
                
            match_data = self.cache[fid_str]
            fixture_meta = self.get_fixture_details(fid)
            
            f_min = {
                "fixture_id": fixture_meta.get("fixture", {}).get("id"),
                "date": fixture_meta.get("fixture", {}).get("date"),
                "home_team": fixture_meta.get("teams", {}).get("home", {}).get("name"),
                "away_team": fixture_meta.get("teams", {}).get("away", {}).get("name"),
                "score_fulltime_home": fixture_meta.get("score", {}).get("fulltime", {}).get("home"),
                "score_fulltime_away": fixture_meta.get("score", {}).get("fulltime", {}).get("away"),
            }
            
            for team_stat in match_data.get("player_statistics", []):
                t_name = team_stat["team"]["name"]
                if t_name.lower() != team_name.lower():
                    continue
                    
                for p_entry in team_stat["players"]:
                    p_id = p_entry["player"]["id"]
                    p_name = p_entry["player"]["name"]
                    
                    for stats in p_entry["statistics"]:
                        minutes = stats["games"]["minutes"]
                        if minutes is None:
                            continue
                        try:
                            minutes = int(minutes)
                        except ValueError:
                            continue
                        
                        if minutes == 0:
                            continue
                        
                        position = stats["games"]["position"]
                        
                        shots_total = stats["shots"]["total"]
                        shots_on = stats["shots"]["on"]
                        goals_total = stats["goals"]["total"]
                        
                        clean_shots_total = int(shots_total) if shots_total is not None else 0
                        clean_shots_on = int(shots_on) if shots_on is not None else 0
                        clean_goals_total = int(goals_total) if goals_total is not None else 0
                        
                        records.append({
                            "fixture_id": f_min["fixture_id"],
                            "fixture_date": f_min["date"],
                            "match_home": f_min["home_team"],
                            "match_away": f_min["away_team"],
                            "fulltime_score_home": f_min["score_fulltime_home"],
                            "fulltime_score_away": f_min["score_fulltime_away"],
                            "player_id": p_id,
                            "player_name": p_name,
                            "games_minutes": minutes,
                            "games_position": position,
                            "shots_total": clean_shots_total,
                            "shots_on": clean_shots_on,
                            "goals_total": clean_goals_total
                        })
                        
        if cache_dirty:
            self._save_cache()
            
        df = pd.DataFrame(records)
        return df
