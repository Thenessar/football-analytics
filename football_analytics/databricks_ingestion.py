import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import requests

from football_analytics.config import BASE_URL, HEADERS, HISTORICAL_ANCHOR_DATE
from football_analytics.config import FIFA_RANKINGS_SEED_AS_OF_DATE, FIFA_RANKINGS_SEED_FILE
from football_analytics.api import FootballApiClient, is_quota_error_payload, payload_hash
from football_analytics.api.exceptions import FootballApiPayloadError
from football_analytics.api.exceptions import FootballApiQuotaError as SharedFootballApiQuotaError
from football_analytics.modeling import build_empirical_bayes_shot_rate_pandas_udf
from football_analytics.quality.validators import (
    SENIOR_MENS_NATIONAL_LEAGUE_IDS,
    ValidationError,
    validate_senior_mens_international_fixture,
)
from football_analytics.storage.delta_io import build_merge_plan

BRONZE_FOOTBALL_MATCH_RAW_PATH = "/mnt/syndicate/bronze/football_match_raw"
BRONZE_FIXTURES_RAW_PATH = "/mnt/syndicate/bronze/football_fixtures_raw"
BRONZE_FIXTURES_ELIGIBILITY_PATH = "/mnt/syndicate/bronze/football_fixture_eligibility"
BRONZE_PLAYER_STATS_RAW_PATH = BRONZE_FOOTBALL_MATCH_RAW_PATH
BRONZE_LINEUPS_RAW_PATH = "/mnt/syndicate/bronze/football_lineups_raw"
SILVER_FIXTURES_PATH = "/mnt/syndicate/silver/football_fixtures"
SILVER_PLAYER_MATCH_STATS_PATH = "/mnt/syndicate/silver/football_player_match_stats"
SILVER_LINEUPS_PATH = "/mnt/syndicate/silver/football_lineups"
GOLD_TEAM_MATCH_CONTEXT_PATH = "/mnt/syndicate/gold/football_team_match_context"
GOLD_RATING_BASELINE_PATH = "/mnt/syndicate/gold/football_rating_baseline"
GOLD_PLAYER_SHOT_FEATURES_PATH = "/mnt/syndicate/gold/football_player_shot_features"
GOLD_PLAYER_SAPM_PATH = "/mnt/syndicate/gold/football_player_sapm"
INGESTION_STATE_CHECKPOINT_TABLE = "default.ingestion_state_checkpoint"
WINDOW_INGESTION_STATE_CHECKPOINT_TABLE = "default.ingestion_window_checkpoint"
DEAD_LETTER_TABLE = "default.football_ingestion_dead_letter"
FIFA_RANKINGS_SEED_TABLE = "default.fifa_mens_world_ranking_seed"
CHECKPOINT_PENDING = "PENDING"
CHECKPOINT_COMPLETED = "COMPLETED"
CHECKPOINT_SKIPPED = "SKIPPED"
CHECKPOINT_FAILED = "FAILED"
CHECKPOINT_QUARANTINED = "QUARANTINED"
COMPLETED_FIXTURE_STATUSES = {"FT", "AET", "PEN"}
FIXTURES_ENDPOINT = "fixtures"
PLAYER_STATS_ENDPOINT = "fixtures/players"
LINEUPS_ENDPOINT = "fixtures/lineups"
QUOTA_ERROR_TOKENS = (
    "rate limit",
    "too many request",
    "too many requests",
    "quota",
    "requests limit",
    "request limit",
    "subscription",
    "exceeded",
)

ACCENTED_CHARS = (
    "ÀÁÂÃÄÅĀĂĄÇĆČÐĎÈÉÊËĒĔĖĘĚÌÍÎÏĪĮİŁÑŃŇÒÓÔÕÖØŌŐŘŚŞŠÙÚÛÜŪŮŰÝŸŽŹŻ"
    "àáâãäåāăąçćčðďèéêëēĕėęěìíîïīįıłñńňòóôõöøōőřśşšùúûüūůűýÿžźż"
)
ASCII_CHARS = (
    "AAAAAAAAACCCDDEEEEEEEEEIIIIIIILNNNOOOOOOOORSSSUUUUUUUYYYZZZ"
    "aaaaaaaaacccddeeeeeeeeeiiiiiilnnnoooooooorsssuuuuuuuyyzzz"
)


class FootballApiQuotaError(SharedFootballApiQuotaError):
    """Raised when API-Football indicates a rate-limit or quota exhaustion event."""


def _require_pyspark():
    try:
        from pyspark.sql import functions as F
        from pyspark.sql.types import (
            ArrayType,
            IntegerType,
            StringType,
            StructField,
            StructType,
        )
    except ImportError as error:
        raise RuntimeError(
            "PySpark is required for Databricks Delta ingestion. Run this module "
            "inside Databricks or install pyspark in the active environment."
        ) from error
    return F, ArrayType, IntegerType, StringType, StructField, StructType


def iter_weekly_windows(
    start_date: str = HISTORICAL_ANCHOR_DATE,
    end_date: Optional[str] = None,
) -> list[tuple[str, str]]:
    """Builds deterministic inclusive 7-day ingestion windows."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date) if end_date else date.today()
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    windows = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=6), end)
        windows.append((cursor.isoformat(), window_end.isoformat()))
        cursor = window_end + timedelta(days=1)
    return windows


def iter_daily_dates(start_date: str, end_date: str) -> list[str]:
    """Builds deterministic inclusive daily load dates."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    days = []
    cursor = start
    while cursor <= end:
        days.append(cursor.isoformat())
        cursor = cursor + timedelta(days=1)
    return days


@dataclass(frozen=True)
class BronzeIngestionSummary:
    requested_dates: tuple[str, ...]
    discovered_fixtures: int
    ingested_fixtures: int
    skipped_fixtures: int
    failed_fixtures: int
    fixture_ids: tuple[int, ...]
    eligible_fixtures: int = 0
    player_stat_payloads_ingested: int = 0
    lineups_ingested: int = 0
    lineups_skipped: int = 0
    quarantined_payloads: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_dates": list(self.requested_dates),
            "discovered_fixtures": self.discovered_fixtures,
            "eligible_fixtures": self.eligible_fixtures or len(self.fixture_ids),
            "ingested_fixtures": self.ingested_fixtures,
            "skipped_fixtures": self.skipped_fixtures,
            "player_stat_payloads_ingested": self.player_stat_payloads_ingested,
            "lineups_ingested": self.lineups_ingested,
            "lineups_skipped": self.lineups_skipped,
            "failed_fixtures": self.failed_fixtures,
            "quarantined_payloads": self.quarantined_payloads,
            "fixture_ids": list(self.fixture_ids),
        }


@dataclass(frozen=True)
class FixtureDiscoveryResult:
    target_date: str
    raw_payload: Mapping
    eligible_fixtures: tuple[Mapping, ...]
    skipped_fixtures: tuple[Mapping, ...]

    @property
    def eligible_fixture_ids(self) -> tuple[int, ...]:
        fixture_ids = []
        for fixture in self.eligible_fixtures:
            fixture_id = (fixture.get("fixture") or {}).get("id")
            if fixture_id is not None:
                fixture_ids.append(int(fixture_id))
        return tuple(fixture_ids)


@dataclass(frozen=True)
class EndpointIngestionPlan:
    fixture_ids_to_fetch: tuple[int, ...]
    skipped_fixture_ids: tuple[int, ...]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def endpoint_ingestion_plan(
    fixture_ids: Iterable[int],
    *,
    completed_fixture_ids: Optional[Iterable[int]] = None,
    force_refresh: bool = False,
) -> EndpointIngestionPlan:
    """Plans endpoint API calls, skipping completed fixtures unless forced."""
    fixture_tuple = tuple(dict.fromkeys(int(fixture_id) for fixture_id in fixture_ids))
    completed = {int(fixture_id) for fixture_id in (completed_fixture_ids or ())}
    if force_refresh:
        return EndpointIngestionPlan(fixture_tuple, ())
    to_fetch = tuple(fixture_id for fixture_id in fixture_tuple if fixture_id not in completed)
    skipped = tuple(fixture_id for fixture_id in fixture_tuple if fixture_id in completed)
    return EndpointIngestionPlan(to_fetch, skipped)


def _supports_spark_sql(spark) -> bool:
    return spark is not None and hasattr(spark, "sql") and hasattr(spark, "createDataFrame")


def split_senior_mens_international_fixtures(
    payload: Mapping,
    *,
    completed_only: bool = True,
) -> tuple[tuple[Mapping, ...], tuple[Mapping, ...]]:
    eligible = []
    skipped = []
    for fixture in payload.get("response", []):
        try:
            validate_senior_mens_international_fixture(
                fixture,
                require_completed=completed_only,
            )
            eligible.append(fixture)
        except ValidationError:
            skipped.append(fixture)
    return tuple(eligible), tuple(skipped)


def is_quota_error_payload(payload: Mapping) -> bool:
    """Detects provider quota/rate-limit messages in API-Football envelopes."""
    errors = payload.get("errors")
    if not errors:
        return False
    text = json.dumps(errors, ensure_ascii=False).casefold()
    return any(token in text for token in QUOTA_ERROR_TOKENS)


def _raise_if_quota_response(response) -> None:
    if getattr(response, "status_code", None) == 429:
        raise FootballApiQuotaError("API-Football HTTP 429: too many requests")


def normalized_name_sql(column_name: str):
    """
    Builds a PySpark Column that lowercases, accent-folds, strips special
    characters, collapses whitespace, and trims names for distributed joins.
    """
    F, *_ = _require_pyspark()
    folded = F.translate(F.lower(F.trim(F.col(column_name))), ACCENTED_CHARS, ASCII_CHARS)
    alphanumeric_spaced = F.regexp_replace(folded, r"[^a-z0-9]+", " ")
    return F.trim(F.regexp_replace(alphanumeric_spaced, r"\s+", " "))


def _football_api_player_stats_schema():
    _, ArrayType, IntegerType, StringType, StructField, StructType = _require_pyspark()
    return StructType([
        StructField("response", ArrayType(StructType([
            StructField("team", StructType([
                StructField("id", IntegerType()),
                StructField("name", StringType()),
            ])),
            StructField("players", ArrayType(StructType([
                StructField("player", StructType([
                    StructField("id", IntegerType()),
                    StructField("name", StringType()),
                ])),
                StructField("statistics", ArrayType(StructType([
                    StructField("games", StructType([
                        StructField("minutes", IntegerType()),
                        StructField("position", StringType()),
                    ])),
                    StructField("shots", StructType([
                        StructField("total", IntegerType()),
                        StructField("on", IntegerType()),
                    ])),
                    StructField("goals", StructType([
                        StructField("total", IntegerType()),
                    ])),
                ]))),
            ]))),
        ]))),
    ])


def _football_api_fixtures_schema():
    _, ArrayType, IntegerType, StringType, StructField, StructType = _require_pyspark()
    return StructType([
        StructField("response", ArrayType(StructType([
            StructField("fixture", StructType([
                StructField("id", IntegerType()),
                StructField("date", StringType()),
                StructField("status", StructType([
                    StructField("short", StringType()),
                    StructField("long", StringType()),
                ])),
                StructField("venue", StructType([
                    StructField("name", StringType()),
                    StructField("city", StringType()),
                ])),
            ])),
            StructField("league", StructType([
                StructField("id", IntegerType()),
                StructField("name", StringType()),
                StructField("season", IntegerType()),
                StructField("country", StringType()),
            ])),
            StructField("teams", StructType([
                StructField("home", StructType([
                    StructField("id", IntegerType()),
                    StructField("name", StringType()),
                ])),
                StructField("away", StructType([
                    StructField("id", IntegerType()),
                    StructField("name", StringType()),
                ])),
            ])),
            StructField("goals", StructType([
                StructField("home", IntegerType()),
                StructField("away", IntegerType()),
            ])),
        ])))]
    )


def _football_api_lineups_schema():
    _, ArrayType, IntegerType, StringType, StructField, StructType = _require_pyspark()
    player_schema = StructType([
        StructField("player", StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
            StructField("number", IntegerType()),
            StructField("pos", StringType()),
        ])),
    ])
    return StructType([
        StructField("response", ArrayType(StructType([
            StructField("team", StructType([
                StructField("id", IntegerType()),
                StructField("name", StringType()),
            ])),
            StructField("formation", StringType()),
            StructField("startXI", ArrayType(player_schema)),
            StructField("substitutes", ArrayType(player_schema)),
        ])))]
    )


def write_player_stats_bronze(
    spark,
    api_payloads: Iterable[Mapping],
    *,
    fixture_id: Optional[int] = None,
    source_endpoint: str = "fixtures/players",
    bronze_path: str = BRONZE_FOOTBALL_MATCH_RAW_PATH,
    mode: str = "append",
) -> None:
    """Persists immutable raw Football-API payloads to the Bronze Delta table."""
    F, *_ = _require_pyspark()
    rows = [
        (
            int(fixture_id) if fixture_id is not None else None,
            source_endpoint,
            json.dumps(payload, ensure_ascii=False),
            payload_hash(payload),
        )
        for payload in api_payloads
    ]
    if not rows:
        return

    bronze_df = (
        spark.createDataFrame(rows, "fixture_id int, source_endpoint string, raw_payload string, response_hash string")
        .withColumn("ingested_at_utc", F.current_timestamp())
    )
    bronze_df.write.format("delta").mode(mode).save(bronze_path)


def _json_payload_rows(
    api_payloads: Iterable[Mapping],
    *,
    run_id: str,
    source_endpoint: str,
    request_params: Mapping,
    target_date: Optional[str] = None,
    fixture_id: Optional[int] = None,
    api_status: str = "OK",
) -> list[tuple]:
    rows = []
    for payload in api_payloads:
        rows.append((
            run_id,
            int(fixture_id) if fixture_id is not None else None,
            source_endpoint,
            json.dumps(dict(request_params), sort_keys=True, ensure_ascii=False),
            target_date,
            json.dumps(payload, ensure_ascii=False),
            payload_hash(payload),
            api_status,
        ))
    return rows


def write_bronze_raw_envelopes(
    spark,
    api_payloads: Iterable[Mapping],
    *,
    run_id: str,
    source_endpoint: str,
    request_params: Mapping,
    target_date: Optional[str] = None,
    fixture_id: Optional[int] = None,
    bronze_path: str,
    mode: str = "append",
) -> None:
    """Persists raw response envelopes with reproducible request metadata."""
    F, *_ = _require_pyspark()
    rows = _json_payload_rows(
        api_payloads,
        run_id=run_id,
        source_endpoint=source_endpoint,
        request_params=request_params,
        target_date=target_date,
        fixture_id=fixture_id,
    )
    if not rows:
        return
    (
        spark.createDataFrame(
            rows,
            (
                "run_id string, fixture_id int, source_endpoint string, "
                "request_params string, target_date string, raw_payload string, "
                "response_hash string, api_status string"
            ),
        )
        .withColumn("target_date", F.to_date("target_date"))
        .withColumn("ingested_at_utc", F.current_timestamp())
        .write
        .format("delta")
        .mode(mode)
        .save(bronze_path)
    )


def write_fixtures_bronze(
    spark,
    payload: Mapping,
    *,
    target_date: str,
    run_id: str,
    bronze_path: str = BRONZE_FIXTURES_RAW_PATH,
) -> None:
    write_bronze_raw_envelopes(
        spark,
        [payload],
        run_id=run_id,
        source_endpoint=FIXTURES_ENDPOINT,
        request_params={"date": target_date, "timezone": "UTC"},
        target_date=target_date,
        bronze_path=bronze_path,
    )


def write_fixture_eligibility_bronze(
    spark,
    *,
    fixtures: Iterable[Mapping],
    target_date: str,
    run_id: str,
    eligibility_status: str,
    bronze_path: str = BRONZE_FIXTURES_ELIGIBILITY_PATH,
) -> None:
    """Persists kept/skipped fixture metadata separately from raw discovery."""
    F, *_ = _require_pyspark()
    rows = []
    for fixture in fixtures:
        fixture_meta = fixture.get("fixture") or {}
        league = fixture.get("league") or {}
        rows.append((
            run_id,
            target_date,
            int(fixture_meta["id"]) if fixture_meta.get("id") is not None else None,
            eligibility_status,
            league.get("id"),
            league.get("name"),
            league.get("season"),
            json.dumps(fixture, ensure_ascii=False),
            payload_hash(fixture),
        ))
    if not rows:
        return
    (
        spark.createDataFrame(
            rows,
            (
                "run_id string, target_date string, fixture_id int, eligibility_status string, "
                "league_id int, league_name string, league_season int, raw_fixture string, response_hash string"
            ),
        )
        .withColumn("target_date", F.to_date("target_date"))
        .withColumn("ingested_at_utc", F.current_timestamp())
        .write
        .format("delta")
        .mode("append")
        .save(bronze_path)
    )


def write_lineups_bronze(
    spark,
    api_payloads: Iterable[Mapping],
    *,
    fixture_id: int,
    run_id: str,
    bronze_path: str = BRONZE_LINEUPS_RAW_PATH,
) -> None:
    write_bronze_raw_envelopes(
        spark,
        api_payloads,
        run_id=run_id,
        source_endpoint=LINEUPS_ENDPOINT,
        request_params={"fixture": int(fixture_id)},
        fixture_id=int(fixture_id),
        bronze_path=bronze_path,
    )


def quarantine_payload(
    spark,
    *,
    payload: Mapping,
    reason: str,
    fixture_id: Optional[int] = None,
    run_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    stage: str = "silver_validate",
    dead_letter_table: str = DEAD_LETTER_TABLE,
) -> None:
    """Writes invalid payloads to a Delta dead-letter table for later inspection."""
    F, *_ = _require_pyspark()
    rows = [(
        run_id,
        int(fixture_id) if fixture_id is not None else None,
        endpoint or stage,
        stage,
        reason,
        json.dumps(payload, ensure_ascii=False),
        payload_hash(payload),
    )]
    (
        spark.createDataFrame(
            rows,
            (
                "run_id string, fixture_id int, endpoint string, stage string, reason string, "
                "raw_payload string, response_hash string"
            ),
        )
        .withColumn("quarantined_at_utc", F.current_timestamp())
        .write
        .format("delta")
        .mode("append")
        .saveAsTable(dead_letter_table)
    )


def validate_or_quarantine_fixture(
    spark,
    fixture: Mapping,
    *,
    dead_letter_table: str = DEAD_LETTER_TABLE,
) -> bool:
    """Returns True for eligible senior men's international fixtures."""
    try:
        validate_senior_mens_international_fixture(fixture)
        return True
    except ValidationError as error:
        quarantine_payload(
            spark,
            payload=fixture,
            reason=str(error),
            fixture_id=(fixture.get("fixture") or {}).get("id"),
            stage="fixture_validation",
            dead_letter_table=dead_letter_table,
        )
        return False


def natural_key_merge_plans() -> dict[str, object]:
    """Documents deterministic Delta MERGE keys used by Databricks tasks."""
    return {
        "fixtures": build_merge_plan(
            "silver.fixtures",
            ("fixture_id",),
            (
                "fixture_id",
                "fixture_date_utc",
                "league_id",
                "league_name",
                "league_season",
                "status_short",
                "response_hash",
                "updated_at_utc",
            ),
        ),
        "player_stats": build_merge_plan(
            "silver.player_stats",
            ("fixture_id", "team_id", "player_id"),
            (
                "fixture_id",
                "team_id",
                "player_id",
                "games_minutes",
                "games_position",
                "shots_total",
                "shots_on",
                "goals_total",
                "response_hash",
                "updated_at_utc",
            ),
        ),
        "lineups": build_merge_plan(
            "silver.lineups",
            ("fixture_id", "team_id", "player_id"),
            (
                "fixture_id",
                "team_id",
                "player_id",
                "player_name",
                "position",
                "number",
                "is_starting",
                "formation",
                "response_hash",
                "updated_at_utc",
            ),
        ),
    }


def build_delta_merge_sql(target: str, source_view: str, keys: Sequence[str], columns: Sequence[str]) -> str:
    if not keys:
        raise ValueError("At least one merge key is required")
    if not columns:
        raise ValueError("At least one merge column is required")
    predicate = " AND ".join(f"target.{key} <=> source.{key}" for key in keys)
    update_columns = [column for column in columns if column not in set(keys)]
    update_clause = ",\n            ".join(
        f"{column} = source.{column}" for column in update_columns
    ) or f"{keys[0]} = source.{keys[0]}"
    insert_columns = ", ".join(columns)
    insert_values = ", ".join(f"source.{column}" for column in columns)
    return f"""
        MERGE INTO {target} AS target
        USING {source_view} AS source
        ON {predicate}
        WHEN MATCHED THEN UPDATE SET
            {update_clause}
        WHEN NOT MATCHED THEN INSERT ({insert_columns})
        VALUES ({insert_values})
    """


def merge_dataframe_to_delta_path(
    spark,
    dataframe,
    *,
    target_path: str,
    keys: Sequence[str],
    temp_view: str,
) -> None:
    """Upserts a DataFrame into a Delta path, bootstrapping the path on first run."""
    try:
        spark.read.format("delta").load(target_path).limit(1).count()
    except Exception:
        dataframe.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
        return

    dataframe.createOrReplaceTempView(temp_view)
    target = f"delta.`{target_path}`"
    spark.sql(build_delta_merge_sql(target, temp_view, keys, tuple(dataframe.columns)))


def fetch_football_api_payload(
    endpoint: str,
    params: Mapping,
    *,
    api_key: Optional[str] = None,
    base_url: str = BASE_URL,
) -> Mapping:
    """Fetches one complete Football-API response envelope for Bronze landing."""
    client = FootballApiClient(
        api_key=api_key,
        base_url=base_url,
        request_get=requests.get,
    )
    try:
        return client.get(endpoint, params)
    except SharedFootballApiQuotaError as error:
        raise FootballApiQuotaError(str(error)) from error
    except FootballApiPayloadError as error:
        raise RuntimeError(str(error).replace("API-Sports", "Football-API")) from error


def write_fifa_rankings_seed_table(
    spark,
    *,
    seed_file: str = FIFA_RANKINGS_SEED_FILE,
    seed_as_of_date: str = FIFA_RANKINGS_SEED_AS_OF_DATE,
    table_name: str = FIFA_RANKINGS_SEED_TABLE,
):
    """
    Materializes the December 2022 FIFA ranking CSV as a typed Delta seed table.

    The source CSV intentionally stays raw in git. This loader normalizes the
    source typo `Raiting` to `rating` for downstream joins and modeling. Rows
    are loaded through Python so Databricks does not treat the repo-relative
    seed file as a Hadoop path.
    """
    rows = [
        {
            **row,
            "ranking_as_of_date": date.fromisoformat(str(row["ranking_as_of_date"])),
        }
        for row in read_fifa_rankings_seed_rows(seed_file, seed_as_of_date=seed_as_of_date)
    ]
    seed_df = spark.createDataFrame(
        rows,
        "rank int, team_name string, rating double, ranking_as_of_date date",
    )
    seed_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
    return seed_df


def resolve_local_seed_file(seed_file: str) -> str:
    """Resolves repo-relative seed paths for local and Databricks bundle runs."""
    path = Path(seed_file)
    if path.is_absolute():
        return str(path)

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return str(cwd_path)

    repo_path = Path(__file__).resolve().parents[1] / path
    if repo_path.exists():
        return str(repo_path)

    return str(path)


def read_fifa_rankings_seed_rows(
    seed_file: str = FIFA_RANKINGS_SEED_FILE,
    *,
    seed_as_of_date: str = FIFA_RANKINGS_SEED_AS_OF_DATE,
) -> list[dict[str, object]]:
    """Reads the raw seed CSV into typed local rows, normalizing `Raiting`."""
    rows = []
    with open(resolve_local_seed_file(seed_file), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "rank": int(row["Rank"]),
                "team_name": row["Team"],
                "rating": float(row["Raiting"]),
                "ranking_as_of_date": seed_as_of_date,
            })
    return rows


def fetch_senior_mens_international_fixtures_for_date(
    match_date: str,
    *,
    api_key: Optional[str] = None,
    completed_only: bool = True,
    allowed_league_ids: set[int] = SENIOR_MENS_NATIONAL_LEAGUE_IDS,
) -> list[Mapping]:
    """Discovers senior men's national-team fixtures for one UTC date."""
    payload = fetch_football_api_payload(
        "fixtures",
        {"date": match_date, "timezone": "UTC"},
        api_key=api_key,
    )
    fixtures = []
    for fixture in payload.get("response", []):
        try:
            validate_senior_mens_international_fixture(
                fixture,
                allowed_league_ids=allowed_league_ids,
                require_completed=completed_only,
            )
        except ValidationError:
            continue
        if completed_only:
            status = ((fixture.get("fixture") or {}).get("status") or {}).get("short")
            if status not in COMPLETED_FIXTURE_STATUSES:
                continue
        fixtures.append(fixture)
    return fixtures


def fetch_international_fixtures_for_date(
    match_date: str,
    *,
    api_key: Optional[str] = None,
    completed_only: bool = True,
    allowed_league_ids: set[int] = SENIOR_MENS_NATIONAL_LEAGUE_IDS,
) -> list[Mapping]:
    """Legacy-neutral alias retained for existing callers."""
    return fetch_senior_mens_international_fixtures_for_date(
        match_date,
        api_key=api_key,
        completed_only=completed_only,
        allowed_league_ids=allowed_league_ids,
    )


def fetch_world_cup_fixtures_for_date(
    match_date: str,
    *,
    api_key: Optional[str] = None,
    completed_only: bool = True,
) -> list[Mapping]:
    """Legacy compatibility wrapper for World Cup-only callers."""
    return fetch_senior_mens_international_fixtures_for_date(
        match_date,
        api_key=api_key,
        completed_only=completed_only,
        allowed_league_ids={1},
    )


def discover_senior_mens_fixtures_for_date(
    spark,
    match_date: str,
    *,
    run_id: str,
    api_key: Optional[str] = None,
    completed_only: bool = True,
    bronze_fixtures_path: str = BRONZE_FIXTURES_RAW_PATH,
    bronze_eligibility_path: str = BRONZE_FIXTURES_ELIGIBILITY_PATH,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
) -> FixtureDiscoveryResult:
    """Calls `/fixtures` once for a date, lands Bronze, and filters eligibility."""
    request_params = {"date": match_date, "timezone": "UTC"}
    upsert_endpoint_checkpoint(
        spark,
        run_id=run_id,
        target_date=match_date,
        fixture_id=None,
        endpoint=FIXTURES_ENDPOINT,
        status=CHECKPOINT_PENDING,
        attempt_count=1,
        checkpoint_table=checkpoint_table,
    )
    try:
        payload = fetch_football_api_payload(
            FIXTURES_ENDPOINT,
            request_params,
            api_key=api_key,
        )
        write_fixtures_bronze(
            spark,
            payload,
            target_date=match_date,
            run_id=run_id,
            bronze_path=bronze_fixtures_path,
        )
        eligible, skipped = split_senior_mens_international_fixtures(
            payload,
            completed_only=completed_only,
        )
        write_fixture_eligibility_bronze(
            spark,
            fixtures=eligible,
            target_date=match_date,
            run_id=run_id,
            eligibility_status="KEPT",
            bronze_path=bronze_eligibility_path,
        )
        write_fixture_eligibility_bronze(
            spark,
            fixtures=skipped,
            target_date=match_date,
            run_id=run_id,
            eligibility_status="SKIPPED",
            bronze_path=bronze_eligibility_path,
        )
        upsert_endpoint_checkpoint(
            spark,
            run_id=run_id,
            target_date=match_date,
            fixture_id=None,
            endpoint=FIXTURES_ENDPOINT,
            status=CHECKPOINT_COMPLETED,
            response_hash=payload_hash(payload),
            checkpoint_table=checkpoint_table,
        )
        return FixtureDiscoveryResult(match_date, payload, eligible, skipped)
    except Exception as error:
        upsert_endpoint_checkpoint(
            spark,
            run_id=run_id,
            target_date=match_date,
            fixture_id=None,
            endpoint=FIXTURES_ENDPOINT,
            status=CHECKPOINT_FAILED,
            last_error=str(error),
            checkpoint_table=checkpoint_table,
        )
        raise


def ingest_player_stats_for_fixtures_to_bronze(
    spark,
    fixture_ids: Iterable[int],
    *,
    api_key: Optional[str] = None,
    bronze_path: str = BRONZE_FOOTBALL_MATCH_RAW_PATH,
    run_id: str = "manual",
    target_date: Optional[str] = None,
    completed_fixture_ids: Optional[Iterable[int]] = None,
    force_refresh: bool = False,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
) -> BronzeIngestionSummary:
    """Fetches `/fixtures/players` for explicit fixture IDs and lands Bronze rows."""
    ingested = []
    failed = 0
    fixture_id_list = [int(fixture_id) for fixture_id in fixture_ids]
    completed_ids = set(completed_fixture_ids or ())
    if completed_fixture_ids is None and _supports_spark_sql(spark):
        completed_ids = completed_checkpoint_fixture_ids(
            spark,
            endpoint=PLAYER_STATS_ENDPOINT,
            target_date=target_date,
            checkpoint_table=checkpoint_table,
        )
    plan = endpoint_ingestion_plan(
        fixture_id_list,
        completed_fixture_ids=completed_ids,
        force_refresh=force_refresh,
    )
    for fixture_id in plan.skipped_fixture_ids:
        if _supports_spark_sql(spark):
            upsert_endpoint_checkpoint(
                spark,
                run_id=run_id,
                target_date=target_date,
                fixture_id=fixture_id,
                endpoint=PLAYER_STATS_ENDPOINT,
                status=CHECKPOINT_SKIPPED,
                checkpoint_table=checkpoint_table,
            )
    for fixture_id in plan.fixture_ids_to_fetch:
        try:
            payload = fetch_football_api_payload(
                PLAYER_STATS_ENDPOINT,
                {"fixture": fixture_id},
                api_key=api_key,
            )
            if _supports_spark_sql(spark):
                write_bronze_raw_envelopes(
                    spark,
                    [payload],
                    run_id=run_id,
                    source_endpoint=PLAYER_STATS_ENDPOINT,
                    request_params={"fixture": fixture_id},
                    fixture_id=fixture_id,
                    bronze_path=bronze_path,
                )
                upsert_endpoint_checkpoint(
                    spark,
                    run_id=run_id,
                    target_date=target_date,
                    fixture_id=fixture_id,
                    endpoint=PLAYER_STATS_ENDPOINT,
                    status=CHECKPOINT_COMPLETED,
                    response_hash=payload_hash(payload),
                    checkpoint_table=checkpoint_table,
                )
            ingested.append(fixture_id)
        except Exception as error:
            if _supports_spark_sql(spark):
                upsert_endpoint_checkpoint(
                    spark,
                    run_id=run_id,
                    target_date=target_date,
                    fixture_id=fixture_id,
                    endpoint=PLAYER_STATS_ENDPOINT,
                    status=CHECKPOINT_FAILED,
                    last_error=str(error),
                    checkpoint_table=checkpoint_table,
                )
            failed += 1
    return BronzeIngestionSummary(
        requested_dates=(),
        discovered_fixtures=len(fixture_id_list),
        ingested_fixtures=len(ingested),
        skipped_fixtures=len(plan.skipped_fixture_ids),
        failed_fixtures=failed,
        fixture_ids=tuple(ingested),
        eligible_fixtures=len(fixture_id_list),
        player_stat_payloads_ingested=len(ingested),
    )


def ingest_lineups_for_fixtures_to_bronze(
    spark,
    fixture_ids: Iterable[int],
    *,
    api_key: Optional[str] = None,
    bronze_path: str = BRONZE_LINEUPS_RAW_PATH,
    run_id: str = "manual",
    target_date: Optional[str] = None,
    completed_fixture_ids: Optional[Iterable[int]] = None,
    force_refresh: bool = False,
    required_fixture_ids: Optional[Iterable[int]] = None,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
) -> BronzeIngestionSummary:
    """Fetches `/fixtures/lineups`, treating historical missing lineups as skips."""
    fixture_id_list = [int(fixture_id) for fixture_id in fixture_ids]
    required = {int(fixture_id) for fixture_id in (required_fixture_ids or ())}
    completed_ids = set(completed_fixture_ids or ())
    if completed_fixture_ids is None and _supports_spark_sql(spark):
        completed_ids = completed_checkpoint_fixture_ids(
            spark,
            endpoint=LINEUPS_ENDPOINT,
            target_date=target_date,
            checkpoint_table=checkpoint_table,
        )
    plan = endpoint_ingestion_plan(
        fixture_id_list,
        completed_fixture_ids=completed_ids,
        force_refresh=force_refresh,
    )
    ingested = []
    skipped = list(plan.skipped_fixture_ids)
    failed = 0
    for fixture_id in plan.fixture_ids_to_fetch:
        try:
            payload = fetch_football_api_payload(
                LINEUPS_ENDPOINT,
                {"fixture": fixture_id},
                api_key=api_key,
            )
            response = payload.get("response", []) if isinstance(payload, Mapping) else []
            if not response:
                skipped.append(fixture_id)
                if _supports_spark_sql(spark):
                    write_lineups_bronze(
                        spark,
                        [payload],
                        fixture_id=fixture_id,
                        run_id=run_id,
                        bronze_path=bronze_path,
                    )
                    upsert_endpoint_checkpoint(
                        spark,
                        run_id=run_id,
                        target_date=target_date,
                        fixture_id=fixture_id,
                        endpoint=LINEUPS_ENDPOINT,
                        status=CHECKPOINT_SKIPPED,
                        response_hash=payload_hash(payload),
                        checkpoint_table=checkpoint_table,
                    )
                continue
            if _supports_spark_sql(spark):
                write_lineups_bronze(
                    spark,
                    [payload],
                    fixture_id=fixture_id,
                    run_id=run_id,
                    bronze_path=bronze_path,
                )
                upsert_endpoint_checkpoint(
                    spark,
                    run_id=run_id,
                    target_date=target_date,
                    fixture_id=fixture_id,
                    endpoint=LINEUPS_ENDPOINT,
                    status=CHECKPOINT_COMPLETED,
                    response_hash=payload_hash(payload),
                    checkpoint_table=checkpoint_table,
                )
            ingested.append(fixture_id)
        except Exception as error:
            failed += 1
            if fixture_id in required:
                raise
            if _supports_spark_sql(spark):
                upsert_endpoint_checkpoint(
                    spark,
                    run_id=run_id,
                    target_date=target_date,
                    fixture_id=fixture_id,
                    endpoint=LINEUPS_ENDPOINT,
                    status=CHECKPOINT_SKIPPED,
                    last_error=str(error),
                    checkpoint_table=checkpoint_table,
                )
            skipped.append(fixture_id)
    return BronzeIngestionSummary(
        requested_dates=(),
        discovered_fixtures=len(fixture_id_list),
        ingested_fixtures=len(ingested),
        skipped_fixtures=len(skipped),
        failed_fixtures=failed,
        fixture_ids=tuple(ingested),
        eligible_fixtures=len(fixture_id_list),
        lineups_ingested=len(ingested),
        lineups_skipped=len(skipped),
    )


def ingest_senior_mens_international_bronze(
    spark,
    *,
    api_key: Optional[str] = None,
    run_id: Optional[str] = None,
    target_date: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    completed_only: bool = True,
    force_refresh: bool = False,
    include_lineups: bool = True,
    required_fixture_ids: Optional[Iterable[int]] = None,
    bronze_fixtures_path: str = BRONZE_FIXTURES_RAW_PATH,
    bronze_eligibility_path: str = BRONZE_FIXTURES_ELIGIBILITY_PATH,
    bronze_player_stats_path: str = BRONZE_FOOTBALL_MATCH_RAW_PATH,
    bronze_lineups_path: str = BRONZE_LINEUPS_RAW_PATH,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
) -> BronzeIngestionSummary:
    """Runs Bronze fixture discovery plus eligible player-stat/lineup ingestion."""
    if target_date:
        dates = [target_date]
    elif date_from and date_to:
        dates = iter_daily_dates(date_from, date_to)
    else:
        raise ValueError("Provide target_date or both date_from and date_to")

    active_run_id = run_id or f"intl-{utc_now_iso()}"
    all_fixture_ids = []
    discovered_count = 0
    skipped_fixture_count = 0
    player_ingested = 0
    lineups_ingested = 0
    lineups_skipped = 0
    failures = 0

    for match_date in dates:
        discovery = discover_senior_mens_fixtures_for_date(
            spark,
            match_date,
            run_id=active_run_id,
            api_key=api_key,
            completed_only=completed_only,
            bronze_fixtures_path=bronze_fixtures_path,
            bronze_eligibility_path=bronze_eligibility_path,
            checkpoint_table=checkpoint_table,
        )
        discovered_count += len(discovery.raw_payload.get("response", []))
        skipped_fixture_count += len(discovery.skipped_fixtures)
        fixture_ids = discovery.eligible_fixture_ids
        all_fixture_ids.extend(fixture_ids)

        player_summary = ingest_player_stats_for_fixtures_to_bronze(
            spark,
            fixture_ids,
            api_key=api_key,
            run_id=active_run_id,
            target_date=match_date,
            force_refresh=force_refresh,
            bronze_path=bronze_player_stats_path,
            checkpoint_table=checkpoint_table,
        )
        player_ingested += player_summary.player_stat_payloads_ingested
        skipped_fixture_count += player_summary.skipped_fixtures
        failures += player_summary.failed_fixtures

        if include_lineups:
            lineup_summary = ingest_lineups_for_fixtures_to_bronze(
                spark,
                fixture_ids,
                api_key=api_key,
                run_id=active_run_id,
                target_date=match_date,
                force_refresh=force_refresh,
                required_fixture_ids=required_fixture_ids,
                bronze_path=bronze_lineups_path,
                checkpoint_table=checkpoint_table,
            )
            lineups_ingested += lineup_summary.lineups_ingested
            lineups_skipped += lineup_summary.lineups_skipped
            failures += lineup_summary.failed_fixtures

    return BronzeIngestionSummary(
        requested_dates=tuple(dates),
        discovered_fixtures=discovered_count,
        eligible_fixtures=len(all_fixture_ids),
        ingested_fixtures=player_ingested,
        skipped_fixtures=skipped_fixture_count,
        failed_fixtures=failures,
        fixture_ids=tuple(all_fixture_ids),
        player_stat_payloads_ingested=player_ingested,
        lineups_ingested=lineups_ingested,
        lineups_skipped=lineups_skipped,
    )


def ingest_senior_mens_international_player_stats_bronze(
    spark,
    *,
    api_key: Optional[str] = None,
    target_date: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    completed_only: bool = True,
    bronze_path: str = BRONZE_FOOTBALL_MATCH_RAW_PATH,
) -> BronzeIngestionSummary:
    """
    Discovers senior men's international fixtures by UTC date/range, then lands player stats in Bronze.

    Daily scheduled runs pass `target_date`. Historical backfills pass
    `date_from` and `date_to`. The downstream Silver transform deduplicates on
    fixture/team/player keys, so reruns are safe for analytical outputs.
    """
    if target_date:
        dates = [target_date]
    elif date_from and date_to:
        dates = iter_daily_dates(date_from, date_to)
    else:
        raise ValueError("Provide target_date or both date_from and date_to")

    discovered = []
    skipped = 0
    for match_date in dates:
        fixtures = fetch_senior_mens_international_fixtures_for_date(
            match_date,
            api_key=api_key,
            completed_only=completed_only,
        )
        for fixture in fixtures:
            fixture_id = (fixture.get("fixture") or {}).get("id")
            if fixture_id:
                discovered.append(int(fixture_id))
            else:
                skipped += 1

    fixture_summary = ingest_player_stats_for_fixtures_to_bronze(
        spark,
        discovered,
        api_key=api_key,
        bronze_path=bronze_path,
    )
    return BronzeIngestionSummary(
        requested_dates=tuple(dates),
        discovered_fixtures=len(discovered),
        ingested_fixtures=fixture_summary.ingested_fixtures,
        skipped_fixtures=skipped + fixture_summary.skipped_fixtures,
        failed_fixtures=fixture_summary.failed_fixtures,
        fixture_ids=fixture_summary.fixture_ids,
    )


def ingest_world_cup_player_stats_bronze(
    spark,
    *,
    api_key: Optional[str] = None,
    target_date: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    completed_only: bool = True,
    bronze_path: str = BRONZE_FOOTBALL_MATCH_RAW_PATH,
) -> BronzeIngestionSummary:
    """Legacy compatibility wrapper for the senior men's international Bronze loader."""
    return ingest_senior_mens_international_player_stats_bronze(
        spark,
        api_key=api_key,
        target_date=target_date,
        date_from=date_from,
        date_to=date_to,
        completed_only=completed_only,
        bronze_path=bronze_path,
    )


def ensure_ingestion_checkpoint_table(
    spark,
    *,
    checkpoint_table: str = WINDOW_INGESTION_STATE_CHECKPOINT_TABLE,
) -> None:
    """Creates the lightweight Delta checkpoint table if it is missing."""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {checkpoint_table} (
            window_start_date DATE,
            window_end_date DATE,
            status STRING,
            updated_timestamp TIMESTAMP,
            records_ingested INT
        )
        USING DELTA
    """)


def seed_ingestion_checkpoint_windows(
    spark,
    windows: Iterable[tuple[str, str]],
    *,
    checkpoint_table: str = WINDOW_INGESTION_STATE_CHECKPOINT_TABLE,
) -> None:
    """Adds PENDING checkpoint rows for windows not already tracked."""
    ensure_ingestion_checkpoint_table(spark, checkpoint_table=checkpoint_table)
    rows = [(start, end, CHECKPOINT_PENDING, 0) for start, end in windows]
    if not rows:
        return

    F, *_ = _require_pyspark()
    staged = (
        spark.createDataFrame(
            rows,
            "window_start_date string, window_end_date string, status string, records_ingested int",
        )
        .withColumn("window_start_date", F.to_date("window_start_date"))
        .withColumn("window_end_date", F.to_date("window_end_date"))
        .withColumn("updated_timestamp", F.current_timestamp())
    )
    staged.createOrReplaceTempView("_pending_ingestion_windows")
    spark.sql(f"""
        MERGE INTO {checkpoint_table} AS target
        USING _pending_ingestion_windows AS source
        ON target.window_start_date = source.window_start_date
           AND target.window_end_date = source.window_end_date
        WHEN NOT MATCHED THEN INSERT *
    """)


def ensure_fixture_endpoint_checkpoint_table(
    spark,
    *,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
) -> None:
    """Creates the Delta endpoint checkpoint/audit table if it is missing."""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {checkpoint_table} (
            run_id STRING,
            target_date DATE,
            fixture_id INT,
            endpoint STRING,
            status STRING,
            attempt_count INT,
            last_error STRING,
            response_hash STRING,
            started_at_utc TIMESTAMP,
            completed_at_utc TIMESTAMP
        )
        USING DELTA
    """)


def checkpoint_merge_sql(
    checkpoint_table: str,
    source_view: str,
) -> str:
    return f"""
        MERGE INTO {checkpoint_table} AS target
        USING {source_view} AS source
        ON target.run_id = source.run_id
           AND target.target_date <=> source.target_date
           AND target.fixture_id <=> source.fixture_id
           AND target.endpoint = source.endpoint
        WHEN MATCHED THEN UPDATE SET
            status = source.status,
            attempt_count = source.attempt_count,
            last_error = source.last_error,
            response_hash = source.response_hash,
            started_at_utc = source.started_at_utc,
            completed_at_utc = source.completed_at_utc
        WHEN NOT MATCHED THEN INSERT *
    """


def upsert_endpoint_checkpoint(
    spark,
    *,
    run_id: str,
    target_date: Optional[str],
    fixture_id: Optional[int],
    endpoint: str,
    status: str,
    attempt_count: int = 1,
    last_error: Optional[str] = None,
    response_hash: Optional[str] = None,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
) -> None:
    ensure_fixture_endpoint_checkpoint_table(spark, checkpoint_table=checkpoint_table)
    F, *_ = _require_pyspark()
    rows = [(
        run_id,
        target_date,
        int(fixture_id) if fixture_id is not None else None,
        endpoint,
        status,
        int(attempt_count),
        last_error,
        response_hash,
    )]
    staged = (
        spark.createDataFrame(
            rows,
            (
                "run_id string, target_date string, fixture_id int, endpoint string, "
                "status string, attempt_count int, last_error string, response_hash string"
            ),
        )
        .withColumn("target_date", F.to_date("target_date"))
        .withColumn("started_at_utc", F.current_timestamp())
        .withColumn(
            "completed_at_utc",
            F.when(F.col("status").isin(CHECKPOINT_COMPLETED, CHECKPOINT_SKIPPED), F.current_timestamp()),
        )
    )
    staged.createOrReplaceTempView("_endpoint_checkpoint_update")
    spark.sql(checkpoint_merge_sql(checkpoint_table, "_endpoint_checkpoint_update"))


def completed_checkpoint_fixture_ids(
    spark,
    *,
    endpoint: str,
    target_date: Optional[str] = None,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
) -> set[int]:
    """Reads completed fixture IDs for endpoint-level rerun skipping."""
    F, *_ = _require_pyspark()
    try:
        checkpoint = spark.table(checkpoint_table)
    except Exception:
        return set()
    filtered = checkpoint.where(
        (F.col("endpoint") == endpoint)
        & (F.col("status") == CHECKPOINT_COMPLETED)
        & F.col("fixture_id").isNotNull()
    )
    if target_date:
        filtered = filtered.where(F.col("target_date") == F.to_date(F.lit(target_date)))
    return {int(row.fixture_id) for row in filtered.select("fixture_id").distinct().collect()}


def update_ingestion_checkpoint(
    spark,
    window_start_date: str,
    window_end_date: str,
    *,
    status: str,
    records_ingested: int,
    checkpoint_table: str = WINDOW_INGESTION_STATE_CHECKPOINT_TABLE,
) -> None:
    """Atomically upserts one checkpoint row after its data commit succeeds."""
    ensure_ingestion_checkpoint_table(spark, checkpoint_table=checkpoint_table)
    row = [(window_start_date, window_end_date, status, int(records_ingested))]
    F, *_ = _require_pyspark()
    staged = (
        spark.createDataFrame(
            row,
            "window_start_date string, window_end_date string, status string, records_ingested int",
        )
        .withColumn("window_start_date", F.to_date("window_start_date"))
        .withColumn("window_end_date", F.to_date("window_end_date"))
        .withColumn("updated_timestamp", F.current_timestamp())
    )
    staged.createOrReplaceTempView("_ingestion_checkpoint_update")
    spark.sql(f"""
        MERGE INTO {checkpoint_table} AS target
        USING _ingestion_checkpoint_update AS source
        ON target.window_start_date = source.window_start_date
           AND target.window_end_date = source.window_end_date
        WHEN MATCHED THEN UPDATE SET
            status = source.status,
            updated_timestamp = source.updated_timestamp,
            records_ingested = source.records_ingested
        WHEN NOT MATCHED THEN INSERT *
    """)


def ingest_fixture_player_stats_to_delta(
    spark,
    fixture_id: int,
    *,
    api_key: Optional[str] = None,
    bronze_path: str = BRONZE_FOOTBALL_MATCH_RAW_PATH,
    silver_path: str = SILVER_PLAYER_MATCH_STATS_PATH,
):
    """
    Fetches a fixture player-stat payload, lands it raw in Bronze, then refreshes Silver.
    """
    payload = fetch_football_api_payload(
        "fixtures/players",
        {"fixture": int(fixture_id)},
        api_key=api_key,
    )
    write_player_stats_bronze(
        spark,
        [payload],
        fixture_id=int(fixture_id),
        bronze_path=bronze_path,
    )
    return transform_bronze_to_silver(
        spark,
        bronze_path=bronze_path,
        silver_path=silver_path,
    )


def transform_bronze_fixtures_to_silver(
    spark,
    *,
    bronze_path: str = BRONZE_FIXTURES_RAW_PATH,
    silver_path: str = SILVER_FIXTURES_PATH,
    mode: str = "merge",
):
    """Normalizes raw `/fixtures` envelopes into the Silver fixtures table."""
    F, *_ = _require_pyspark()
    schema = _football_api_fixtures_schema()
    bronze = spark.read.format("delta").load(bronze_path)
    parsed = bronze.withColumn("payload", F.from_json(F.col("raw_payload"), schema))
    flattened = parsed.withColumn("fixture_entry", F.explode_outer("payload.response"))
    silver = (
        flattened
        .select(
            F.col("fixture_entry.fixture.id").cast("int").alias("fixture_id"),
            F.to_timestamp("fixture_entry.fixture.date").alias("fixture_date_utc"),
            F.col("fixture_entry.league.id").cast("int").alias("league_id"),
            F.col("fixture_entry.league.name").alias("league_name"),
            F.col("fixture_entry.league.season").cast("int").alias("league_season"),
            F.col("fixture_entry.teams.home.id").cast("int").alias("home_team_id"),
            F.col("fixture_entry.teams.home.name").alias("home_team_name"),
            F.col("fixture_entry.teams.away.id").cast("int").alias("away_team_id"),
            F.col("fixture_entry.teams.away.name").alias("away_team_name"),
            F.col("fixture_entry.goals.home").cast("int").alias("home_goals"),
            F.col("fixture_entry.goals.away").cast("int").alias("away_goals"),
            F.col("fixture_entry.fixture.status.short").alias("status_short"),
            F.col("fixture_entry.fixture.status.long").alias("status_long"),
            F.col("fixture_entry.fixture.venue.name").alias("venue"),
            F.col("fixture_entry.league.country").alias("country"),
            F.col("response_hash"),
        )
        .where(F.col("fixture_id").isNotNull())
        .withColumn("updated_at_utc", F.current_timestamp())
        .dropDuplicates(["fixture_id"])
    )
    if mode == "merge":
        merge_dataframe_to_delta_path(
            spark,
            silver,
            target_path=silver_path,
            keys=("fixture_id",),
            temp_view="_silver_fixtures_updates",
        )
    else:
        silver.write.format("delta").mode(mode).option("overwriteSchema", "true").save(silver_path)
    return silver


def transform_bronze_to_silver(
    spark,
    *,
    bronze_path: str = BRONZE_FOOTBALL_MATCH_RAW_PATH,
    silver_path: str = SILVER_PLAYER_MATCH_STATS_PATH,
    mode: str = "merge",
):
    """
    Cleans raw Football-API player-stat payloads into typed Silver rows.

    Nulls in shots.total, shots.on, and goals.total are intercepted at the
    nested metric node and explicitly cast to integer zero.
    """
    F, *_ = _require_pyspark()
    schema = _football_api_player_stats_schema()

    bronze = spark.read.format("delta").load(bronze_path)
    parsed = bronze.withColumn("payload", F.from_json(F.col("raw_payload"), schema))
    flattened = (
        parsed
        .withColumn("team_entry", F.explode_outer("payload.response"))
        .withColumn("player_entry", F.explode_outer("team_entry.players"))
        .withColumn("stat_entry", F.explode_outer("player_entry.statistics"))
    )

    silver = (
        flattened
        .select(
            F.col("fixture_id").cast("int").alias("fixture_id"),
            F.col("team_entry.team.id").cast("int").alias("team_id"),
            F.col("team_entry.team.name").alias("team_name"),
            F.col("player_entry.player.id").cast("int").alias("player_id"),
            F.col("player_entry.player.name").alias("player_name"),
            F.col("stat_entry.games.minutes").cast("int").alias("games_minutes"),
            F.col("stat_entry.games.position").alias("games_position"),
            F.coalesce(F.col("stat_entry.shots.total"), F.lit(0)).cast("int").alias("shots_total"),
            F.coalesce(F.col("stat_entry.shots.on"), F.lit(0)).cast("int").alias("shots_on"),
            F.coalesce(F.col("stat_entry.goals.total"), F.lit(0)).cast("int").alias("goals_total"),
            F.col("response_hash"),
        )
        .where(F.col("player_id").isNotNull())
        .withColumn("player_name_normalized", normalized_name_sql("player_name"))
        .withColumn("team_name_normalized", normalized_name_sql("team_name"))
        .withColumn("updated_at_utc", F.current_timestamp())
        .dropDuplicates(["fixture_id", "team_id", "player_id"])
    )

    if mode == "merge":
        merge_dataframe_to_delta_path(
            spark,
            silver,
            target_path=silver_path,
            keys=("fixture_id", "team_id", "player_id"),
            temp_view="_silver_player_stats_updates",
        )
    else:
        silver.write.format("delta").mode(mode).option("overwriteSchema", "true").save(silver_path)
    return silver


def transform_bronze_lineups_to_silver(
    spark,
    *,
    bronze_path: str = BRONZE_LINEUPS_RAW_PATH,
    silver_path: str = SILVER_LINEUPS_PATH,
    mode: str = "merge",
):
    """Normalizes raw `/fixtures/lineups` envelopes into Silver lineup rows."""
    F, *_ = _require_pyspark()
    schema = _football_api_lineups_schema()
    bronze = spark.read.format("delta").load(bronze_path)
    parsed = bronze.withColumn("payload", F.from_json(F.col("raw_payload"), schema))
    team_rows = parsed.withColumn("team_entry", F.explode_outer("payload.response"))
    starters = (
        team_rows
        .withColumn("player_entry", F.explode_outer("team_entry.startXI"))
        .withColumn("is_starting", F.lit(True))
    )
    substitutes = (
        team_rows
        .withColumn("player_entry", F.explode_outer("team_entry.substitutes"))
        .withColumn("is_starting", F.lit(False))
    )
    flattened = starters.unionByName(substitutes, allowMissingColumns=True)
    silver = (
        flattened
        .select(
            F.col("fixture_id").cast("int").alias("fixture_id"),
            F.col("team_entry.team.id").cast("int").alias("team_id"),
            F.col("team_entry.team.name").alias("team_name"),
            F.col("player_entry.player.id").cast("int").alias("player_id"),
            F.col("player_entry.player.name").alias("player_name"),
            F.col("player_entry.player.pos").alias("position"),
            F.col("player_entry.player.number").cast("int").alias("number"),
            F.col("is_starting"),
            F.col("team_entry.formation").alias("formation"),
            F.col("response_hash"),
        )
        .where(F.col("player_id").isNotNull())
        .withColumn("updated_at_utc", F.current_timestamp())
        .dropDuplicates(["fixture_id", "team_id", "player_id"])
    )
    if mode == "merge":
        merge_dataframe_to_delta_path(
            spark,
            silver,
            target_path=silver_path,
            keys=("fixture_id", "team_id", "player_id"),
            temp_view="_silver_lineups_updates",
        )
    else:
        silver.write.format("delta").mode(mode).option("overwriteSchema", "true").save(silver_path)
    return silver


def build_gold_team_match_context(
    spark,
    *,
    silver_fixtures_path: str = SILVER_FIXTURES_PATH,
    gold_path: str = GOLD_TEAM_MATCH_CONTEXT_PATH,
    mode: str = "overwrite",
):
    """Builds team-level match context features from Silver fixtures."""
    F, *_ = _require_pyspark()
    fixtures = spark.read.format("delta").load(silver_fixtures_path)
    home_rows = fixtures.select(
        "fixture_id",
        "fixture_date_utc",
        F.col("home_team_id").alias("team_id"),
        F.col("home_team_name").alias("team_name"),
        F.col("away_team_id").alias("opponent_team_id"),
        F.col("away_team_name").alias("opponent_team_name"),
        F.lit("home").alias("home_away"),
        F.col("home_goals").alias("goals_for"),
        F.col("away_goals").alias("goals_against"),
        "league_id",
        "league_name",
        "league_season",
        "status_short",
    )
    away_rows = fixtures.select(
        "fixture_id",
        "fixture_date_utc",
        F.col("away_team_id").alias("team_id"),
        F.col("away_team_name").alias("team_name"),
        F.col("home_team_id").alias("opponent_team_id"),
        F.col("home_team_name").alias("opponent_team_name"),
        F.lit("away").alias("home_away"),
        F.col("away_goals").alias("goals_for"),
        F.col("home_goals").alias("goals_against"),
        "league_id",
        "league_name",
        "league_season",
        "status_short",
    )
    gold = (
        home_rows
        .unionByName(away_rows)
        .withColumn("team_name_normalized", normalized_name_sql("team_name"))
        .withColumn("opponent_team_name_normalized", normalized_name_sql("opponent_team_name"))
        .withColumn("updated_at_utc", F.current_timestamp())
    )
    gold.write.format("delta").mode(mode).option("overwriteSchema", "true").save(gold_path)
    return gold


def build_gold_rating_baseline(
    spark,
    *,
    seed_table: str = FIFA_RANKINGS_SEED_TABLE,
    gold_path: str = GOLD_RATING_BASELINE_PATH,
    mode: str = "overwrite",
):
    """Builds the model-ready rating baseline from the typed FIFA seed table."""
    F, *_ = _require_pyspark()
    baseline = (
        spark.table(seed_table)
        .select("rank", "team_name", "rating", "ranking_as_of_date")
        .withColumn("team_name_normalized", normalized_name_sql("team_name"))
        .withColumn("updated_at_utc", F.current_timestamp())
    )
    baseline.write.format("delta").mode(mode).option("overwriteSchema", "true").save(gold_path)
    return baseline


def build_gold_player_shot_features(
    spark,
    *,
    silver_player_stats_path: str = SILVER_PLAYER_MATCH_STATS_PATH,
    silver_fixtures_path: str = SILVER_FIXTURES_PATH,
    gold_path: str = GOLD_PLAYER_SHOT_FEATURES_PATH,
    mode: str = "overwrite",
):
    """Builds model-ready player shot features from Silver player stats and fixtures."""
    F, *_ = _require_pyspark()
    players = spark.read.format("delta").load(silver_player_stats_path)
    fixtures = spark.read.format("delta").load(silver_fixtures_path).select(
        "fixture_id",
        "fixture_date_utc",
        "league_id",
        "league_name",
        "league_season",
        "home_team_id",
        "away_team_id",
    )
    features = (
        players
        .join(fixtures, "fixture_id", "left")
        .withColumn(
            "opponent_team_id",
            F.when(F.col("team_id") == F.col("home_team_id"), F.col("away_team_id"))
            .when(F.col("team_id") == F.col("away_team_id"), F.col("home_team_id")),
        )
        .withColumn(
            "shots_per_90",
            F.when(F.col("games_minutes") > 0, F.col("shots_total") * F.lit(90.0) / F.col("games_minutes"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "shots_on_per_90",
            F.when(F.col("games_minutes") > 0, F.col("shots_on") * F.lit(90.0) / F.col("games_minutes"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn("updated_at_utc", F.current_timestamp())
    )
    features.write.format("delta").mode(mode).option("overwriteSchema", "true").save(gold_path)
    return features


def transform_silver_to_gold_sapm(
    spark,
    *,
    silver_path: str = SILVER_PLAYER_MATCH_STATS_PATH,
    position_prior_path: Optional[str] = None,
    fixture_context_path: Optional[str] = None,
    gold_path: str = GOLD_PLAYER_SAPM_PATH,
    mode: str = "overwrite",
):
    """
    Builds Gold-ready S-APM features from Silver player-match rows.

    Optional context Delta inputs can provide position-level alpha/beta priors
    and fixture-level stake/opponent weights. Missing context defaults to
    neutral priors and weights so the transform remains deterministic.
    """
    F, *_ = _require_pyspark()
    smoothed_shot_rate_udf = build_empirical_bayes_shot_rate_pandas_udf()

    silver = spark.read.format("delta").load(silver_path)
    enriched = silver.withColumn(
        "position_group",
        F.when(F.upper(F.col("games_position")).isin("F", "A", "ATTACKER", "FORWARD", "STRIKER"), F.lit("F"))
        .when(F.upper(F.col("games_position")).isin("D", "DEFENDER", "DEFENCE"), F.lit("D"))
        .when(F.upper(F.col("games_position")).isin("G", "GK", "GOALKEEPER"), F.lit("G"))
        .otherwise(F.lit("M")),
    )

    if position_prior_path:
        priors = spark.read.format("delta").load(position_prior_path).select(
            "position_group",
            F.col("alpha").cast("double").alias("prior_alpha"),
            F.col("beta").cast("double").alias("prior_beta"),
        )
        enriched = enriched.join(priors, "position_group", "left")
    else:
        enriched = (
            enriched
            .withColumn(
                "prior_alpha",
                F.when(F.col("position_group") == "F", F.lit(2.6 * 3.0))
                .when(F.col("position_group") == "M", F.lit(1.2 * 3.0))
                .when(F.col("position_group") == "D", F.lit(0.5 * 3.0))
                .otherwise(F.lit(0.0)),
            )
            .withColumn("prior_beta", F.lit(270.0))
        )

    if fixture_context_path:
        context = spark.read.format("delta").load(fixture_context_path).select(
            F.col("fixture_id").cast("int").alias("context_fixture_id"),
            F.col("game_importance_scalar").cast("double"),
            F.col("opponent_strength_adjustment").cast("double"),
            F.col("defensive_containment_rating").cast("double"),
            F.col("defensive_elo").cast("double"),
        )
        enriched = enriched.join(
            context,
            enriched.fixture_id == context.context_fixture_id,
            "left",
        ).drop("context_fixture_id")
    else:
        enriched = (
            enriched
            .withColumn("game_importance_scalar", F.lit(1.0))
            .withColumn("opponent_strength_adjustment", F.lit(1.0))
            .withColumn("defensive_containment_rating", F.lit(1.0))
            .withColumn("defensive_elo", F.lit(1500.0))
        )

    gold = (
        enriched
        .withColumn("prior_alpha", F.coalesce(F.col("prior_alpha"), F.lit(0.0)))
        .withColumn("prior_beta", F.coalesce(F.col("prior_beta"), F.lit(270.0)))
        .withColumn("game_importance_scalar", F.coalesce(F.col("game_importance_scalar"), F.lit(1.0)))
        .withColumn("opponent_strength_adjustment", F.coalesce(F.col("opponent_strength_adjustment"), F.lit(1.0)))
        .withColumn("shot_rate_smoothed_per_minute", smoothed_shot_rate_udf(
            F.col("shots_total"),
            F.col("games_minutes"),
            F.col("prior_alpha"),
            F.col("prior_beta"),
        ))
        .withColumn(
            "sapm_interaction_weight",
            F.col("game_importance_scalar") * F.col("opponent_strength_adjustment"),
        )
        .withColumn(
            "weighted_shots",
            F.col("sapm_interaction_weight") * F.col("shots_total"),
        )
        .withColumn(
            "weighted_minutes",
            F.col("sapm_interaction_weight") * F.col("games_minutes"),
        )
    )

    gold.write.format("delta").mode(mode).save(gold_path)
    return gold
