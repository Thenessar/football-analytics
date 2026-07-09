import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Mapping, Optional, Sequence

import requests

from football_analytics.config import BASE_URL, HEADERS, HISTORICAL_ANCHOR_DATE
from football_analytics.api import FootballApiClient, is_quota_error_payload, payload_hash
from football_analytics.api.exceptions import FootballApiPayloadError
from football_analytics.api.exceptions import FootballApiQuotaError as SharedFootballApiQuotaError
from football_analytics.quality.validators import (
    SENIOR_MENS_NATIONAL_LEAGUE_IDS,
    ValidationError,
    validate_senior_mens_international_fixture,
)

BRONZE_FOOTBALL_MATCH_RAW_PATH = "/mnt/syndicate/bronze/football_match_raw"
BRONZE_FIXTURES_RAW_PATH = "/mnt/syndicate/bronze/football_fixtures_raw"
BRONZE_FIXTURES_ELIGIBILITY_PATH = "/mnt/syndicate/bronze/football_fixture_eligibility"
BRONZE_PLAYER_STATS_RAW_PATH = BRONZE_FOOTBALL_MATCH_RAW_PATH
BRONZE_LINEUPS_RAW_PATH = "/mnt/syndicate/bronze/football_lineups_raw"
BRONZE_FIXTURE_STATISTICS_RAW_PATH = "/mnt/syndicate/bronze/football_fixture_statistics_raw"
INGESTION_STATE_CHECKPOINT_TABLE = "default.bronze_ingestion_state_checkpoint"
CHECKPOINT_PENDING = "PENDING"
CHECKPOINT_COMPLETED = "COMPLETED"
CHECKPOINT_SKIPPED = "SKIPPED"
CHECKPOINT_FAILED = "FAILED"
COMPLETED_FIXTURE_STATUSES = {"FT", "AET", "PEN"}
LIVE_FIXTURE_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT"}
TERMINAL_UNPLAYED_FIXTURE_STATUSES = {"PST", "CANC", "ABD", "AWD", "WO"}
FIXTURE_STATE_SCHEDULED = "SCHEDULED"
FIXTURE_STATE_LINEUPS_CONFIRMED = "LINEUPS_CONFIRMED"
FIXTURE_STATE_LIVE = "LIVE"
FIXTURE_STATE_COMPLETED = "COMPLETED"
DEFAULT_LOOKAHEAD_DAYS = 7
DEFAULT_LOOKBACK_DAYS = 2
FIXTURES_ENDPOINT = "fixtures"
PLAYER_STATS_ENDPOINT = "fixtures/players"
LINEUPS_ENDPOINT = "fixtures/lineups"
STATISTICS_ENDPOINT = "fixtures/statistics"
DEFAULT_API_RATE_LIMIT_PER_MINUTE = 240
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

def is_delta_table_target(target: str) -> bool:
    """Returns True when a Delta target is a catalog table, not a filesystem path."""
    return not (
        target.startswith("/")
        or target.startswith("dbfs:")
        or "://" in target
        or target.startswith("delta.`")
    )


def read_delta_target(spark, target: str):
    if is_delta_table_target(target):
        return spark.table(target)
    return spark.read.format("delta").load(target)


def delta_target_exists(spark, target: str) -> bool:
    if is_delta_table_target(target):
        return bool(spark.catalog.tableExists(target))
    try:
        spark.read.format("delta").load(target).limit(1).count()
        return True
    except Exception:
        return False


def write_delta_target(dataframe, target: str, *, mode: str, overwrite_schema: bool = False) -> None:
    writer = dataframe.write.format("delta").mode(mode)
    if overwrite_schema:
        writer = writer.option("overwriteSchema", "true")
    if is_delta_table_target(target):
        writer.saveAsTable(target)
    else:
        writer.save(target)


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


@dataclass(frozen=True)
class IngestionDateSelection:
    dates: tuple[str, ...]
    mode: str
    defaulted_from_checkpoint: bool = False


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


def resolve_ingestion_dates(
    spark,
    *,
    target_date: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
    today: Optional[str] = None,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> IngestionDateSelection:
    """Resolves manual date overrides or the default rolling ingestion range."""
    if target_date:
        return IngestionDateSelection((target_date,), "target_date")
    if date_from or date_to:
        if not (date_from and date_to):
            raise ValueError("Provide both date_from and date_to, or neither")
        return IngestionDateSelection(tuple(iter_daily_dates(date_from, date_to)), "date_range")

    today_date = date.fromisoformat(today) if today else date.today()
    lookahead_days = max(0, int(lookahead_days))
    lookback_days = max(0, int(lookback_days))
    lookback_start = today_date - timedelta(days=lookback_days)
    end_date = (today_date + timedelta(days=lookahead_days)).isoformat()
    checkpoint_start = latest_completed_checkpoint_date(
        spark,
        checkpoint_table=checkpoint_table,
        max_target_date=today_date.isoformat(),
    ) or today_date.isoformat()
    start_date = min(date.fromisoformat(checkpoint_start), lookback_start).isoformat()
    if date.fromisoformat(start_date) > date.fromisoformat(end_date):
        start_date = end_date
    return IngestionDateSelection(
        tuple(iter_daily_dates(start_date, end_date)),
        "checkpoint_to_lookahead",
        defaulted_from_checkpoint=True,
    )


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
    statistics_ingested: int = 0
    statistics_skipped: int = 0
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
            "statistics_ingested": self.statistics_ingested,
            "statistics_skipped": self.statistics_skipped,
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


@dataclass(frozen=True)
class FixtureEndpointFetchResult:
    index: int
    fixture_id: int
    payload: Optional[Mapping] = None
    error: Optional[Exception] = None


class ApiRateLimiter:
    """Thread-safe spacing limiter for API calls from the Databricks driver."""

    def __init__(self, calls_per_minute: Optional[int]):
        self._interval_seconds = 0.0
        if calls_per_minute and calls_per_minute > 0:
            self._interval_seconds = 60.0 / float(calls_per_minute)
        self._lock = threading.Lock()
        self._next_available_at = 0.0

    def wait(self) -> None:
        if self._interval_seconds <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_available_at - now)
            self._next_available_at = max(now, self._next_available_at) + self._interval_seconds
        if wait_seconds > 0:
            time.sleep(wait_seconds)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fixture_status_short(fixture: Mapping) -> Optional[str]:
    status = ((fixture.get("fixture") or {}).get("status") or {}).get("short")
    return str(status).upper() if status else None


def lineups_payload_has_confirmed_starting_xi(payload: Optional[Mapping]) -> bool:
    if not isinstance(payload, Mapping):
        return False
    response = payload.get("response")
    if not isinstance(response, list) or len(response) < 2:
        return False
    confirmed_teams = 0
    for team_lineup in response:
        start_xi = team_lineup.get("startXI") if isinstance(team_lineup, Mapping) else None
        if isinstance(start_xi, list) and len(start_xi) >= 11:
            confirmed_teams += 1
    return confirmed_teams >= 2


def player_stats_payload_has_match_data(payload: Optional[Mapping]) -> bool:
    if not isinstance(payload, Mapping):
        return False
    response = payload.get("response")
    if not isinstance(response, list) or len(response) < 2:
        return False
    return any(bool(team.get("players")) for team in response if isinstance(team, Mapping))


def fixture_statistics_payload_has_data(payload: Optional[Mapping]) -> bool:
    if not isinstance(payload, Mapping):
        return False
    response = payload.get("response")
    if not isinstance(response, list) or len(response) < 2:
        return False
    return any(bool(team.get("statistics")) for team in response if isinstance(team, Mapping))


def classify_fixture_business_state(
    fixture: Mapping,
    *,
    lineups_payload: Optional[Mapping] = None,
    player_stats_payload: Optional[Mapping] = None,
) -> str:
    """Maps provider status and endpoint payloads to the business fixture lifecycle."""
    status = fixture_status_short(fixture)
    if status in COMPLETED_FIXTURE_STATUSES and player_stats_payload_has_match_data(player_stats_payload):
        return FIXTURE_STATE_COMPLETED
    if status in LIVE_FIXTURE_STATUSES:
        return FIXTURE_STATE_LIVE
    if lineups_payload_has_confirmed_starting_xi(lineups_payload):
        return FIXTURE_STATE_LINEUPS_CONFIRMED
    if status in COMPLETED_FIXTURE_STATUSES:
        return FIXTURE_STATE_LIVE
    return FIXTURE_STATE_SCHEDULED


def fixture_ids_for_player_stats(fixtures: Iterable[Mapping]) -> tuple[int, ...]:
    fixture_ids = []
    for fixture in fixtures:
        status = fixture_status_short(fixture)
        if status in COMPLETED_FIXTURE_STATUSES:
            fixture_id = (fixture.get("fixture") or {}).get("id")
            if fixture_id is not None:
                fixture_ids.append(int(fixture_id))
    return tuple(dict.fromkeys(fixture_ids))


def fixture_ids_for_fixture_statistics(fixtures: Iterable[Mapping]) -> tuple[int, ...]:
    """Team statistics load only for completed fixtures, same rule as player stats."""
    return fixture_ids_for_player_stats(fixtures)


def fixture_ids_for_lineups(fixtures: Iterable[Mapping]) -> tuple[int, ...]:
    fixture_ids = []
    for fixture in fixtures:
        status = fixture_status_short(fixture)
        if status not in TERMINAL_UNPLAYED_FIXTURE_STATUSES:
            fixture_id = (fixture.get("fixture") or {}).get("id")
            if fixture_id is not None:
                fixture_ids.append(int(fixture_id))
    return tuple(dict.fromkeys(fixture_ids))


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


def _safe_log_error(error: Exception) -> str:
    return " ".join(str(error).split())[:500]


def _log_ingestion_event(logger: Optional[logging.Logger], level: int, event: str, **fields) -> None:
    if logger is None:
        return
    logger.log(level, event, extra={"event": event, **fields})


def _date_scope_log_fields(dates: Sequence[str]) -> dict[str, object]:
    if len(dates) == 1:
        return {
            "date_scope": "single_day",
            "target_date": dates[0],
            "requested_dates_count": 1,
        }
    return {
        "date_scope": "date_range",
        "date_from": dates[0],
        "date_to": dates[-1],
        "requested_dates_count": len(dates),
    }


def _endpoint_worker_count(endpoint_max_workers: int, fixture_count: int) -> int:
    if fixture_count <= 0:
        return 0
    try:
        requested_workers = int(endpoint_max_workers)
    except (TypeError, ValueError):
        requested_workers = 1
    return max(1, min(requested_workers, fixture_count))


def _mark_fixture_endpoint_started(
    *,
    run_id: str,
    target_date: Optional[str],
    endpoint: str,
    fixture_id: int,
    index: int,
    total: int,
    logger: Optional[logging.Logger],
) -> None:
    _log_ingestion_event(
        logger,
        logging.INFO,
        "fixture_endpoint_started",
        run_id=run_id,
        stage="bronze_ingest",
        target_date=target_date,
        endpoint=endpoint,
        fixture_id=fixture_id,
        index=index,
        total=total,
        status=CHECKPOINT_PENDING,
    )


def _mark_fixture_endpoints_pending(
    spark,
    fixture_ids: Sequence[int],
    *,
    run_id: str,
    target_date: Optional[str],
    endpoint: str,
    checkpoint_table: str,
    logger: Optional[logging.Logger],
) -> None:
    total = len(fixture_ids)
    for index, fixture_id in enumerate(fixture_ids, start=1):
        _mark_fixture_endpoint_started(
            run_id=run_id,
            target_date=target_date,
            endpoint=endpoint,
            fixture_id=fixture_id,
            index=index,
            total=total,
            logger=logger,
        )
    if _supports_spark_sql(spark):
        upsert_endpoint_checkpoints(
            spark,
            [
                {
                    "run_id": run_id,
                    "target_date": target_date,
                    "fixture_id": fixture_id,
                    "endpoint": endpoint,
                    "status": CHECKPOINT_PENDING,
                }
                for fixture_id in fixture_ids
            ],
            checkpoint_table=checkpoint_table,
        )


def _fetch_payload_with_optional_logger(
    endpoint: str,
    params: Mapping,
    *,
    api_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Mapping:
    if logger is None:
        return fetch_football_api_payload(endpoint, params, api_key=api_key)
    return fetch_football_api_payload(endpoint, params, api_key=api_key, logger=logger)


def _fetch_fixture_endpoint_result(
    *,
    endpoint: str,
    fixture_id: int,
    index: int,
    api_key: Optional[str],
    logger: Optional[logging.Logger],
    rate_limiter: ApiRateLimiter,
) -> FixtureEndpointFetchResult:
    try:
        rate_limiter.wait()
        payload = _fetch_payload_with_optional_logger(
            endpoint,
            {"fixture": fixture_id},
            api_key=api_key,
            logger=logger,
        )
        return FixtureEndpointFetchResult(index=index, fixture_id=fixture_id, payload=payload)
    except Exception as error:
        return FixtureEndpointFetchResult(index=index, fixture_id=fixture_id, error=error)


def _iter_fixture_endpoint_fetch_results(
    spark,
    fixture_ids: Sequence[int],
    *,
    endpoint: str,
    api_key: Optional[str],
    run_id: str,
    target_date: Optional[str],
    checkpoint_table: str,
    logger: Optional[logging.Logger],
    endpoint_max_workers: int,
    api_rate_limit_per_minute: Optional[int],
) -> Iterable[FixtureEndpointFetchResult]:
    """Yields endpoint fetch results while keeping Spark writes on the driver."""
    fixture_id_tuple = tuple(int(fixture_id) for fixture_id in fixture_ids)
    total = len(fixture_id_tuple)
    workers = _endpoint_worker_count(endpoint_max_workers, total)
    if total == 0:
        return

    started_at = time.monotonic()
    successful_fetches = 0
    failed_fetches = 0
    rate_limiter = ApiRateLimiter(api_rate_limit_per_minute)
    if workers == 1:
        _mark_fixture_endpoints_pending(
            spark,
            fixture_id_tuple,
            run_id=run_id,
            target_date=target_date,
            endpoint=endpoint,
            checkpoint_table=checkpoint_table,
            logger=logger,
        )
        for index, fixture_id in enumerate(fixture_id_tuple, start=1):
            result = _fetch_fixture_endpoint_result(
                endpoint=endpoint,
                fixture_id=fixture_id,
                index=index,
                api_key=api_key,
                logger=logger,
                rate_limiter=rate_limiter,
            )
            if result.error is None:
                successful_fetches += 1
            else:
                failed_fetches += 1
            yield result
        duration_seconds = max(time.monotonic() - started_at, 0.0)
        _log_ingestion_event(
            logger,
            logging.INFO,
            "endpoint_fetch_batch_completed",
            run_id=run_id,
            stage="bronze_ingest",
            target_date=target_date,
            endpoint=endpoint,
            total=total,
            successful_fetches=successful_fetches,
            failed_fetches=failed_fetches,
            duration_seconds=round(duration_seconds, 3),
            actual_fetches_per_minute=round((total / duration_seconds) * 60, 2) if duration_seconds > 0 else None,
            endpoint_max_workers=workers,
            api_rate_limit_per_minute=api_rate_limit_per_minute,
        )
        return

    _log_ingestion_event(
        logger,
        logging.INFO,
        "endpoint_parallel_fetch_started",
        run_id=run_id,
        stage="bronze_ingest",
        target_date=target_date,
        endpoint=endpoint,
        total=total,
        endpoint_max_workers=workers,
        api_rate_limit_per_minute=api_rate_limit_per_minute,
    )
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bronze-api-fetch") as executor:
        _mark_fixture_endpoints_pending(
            spark,
            fixture_id_tuple,
            run_id=run_id,
            target_date=target_date,
            endpoint=endpoint,
            checkpoint_table=checkpoint_table,
            logger=logger,
        )
        futures = []
        for index, fixture_id in enumerate(fixture_id_tuple, start=1):
            futures.append(executor.submit(
                _fetch_fixture_endpoint_result,
                endpoint=endpoint,
                fixture_id=fixture_id,
                index=index,
                api_key=api_key,
                logger=logger,
                rate_limiter=rate_limiter,
            ))
        for future in as_completed(futures):
            result = future.result()
            if result.error is None:
                successful_fetches += 1
            else:
                failed_fetches += 1
            yield result
    duration_seconds = max(time.monotonic() - started_at, 0.0)
    _log_ingestion_event(
        logger,
        logging.INFO,
        "endpoint_fetch_batch_completed",
        run_id=run_id,
        stage="bronze_ingest",
        target_date=target_date,
        endpoint=endpoint,
        total=total,
        successful_fetches=successful_fetches,
        failed_fetches=failed_fetches,
        duration_seconds=round(duration_seconds, 3),
        actual_fetches_per_minute=round((total / duration_seconds) * 60, 2) if duration_seconds > 0 else None,
        endpoint_max_workers=workers,
        api_rate_limit_per_minute=api_rate_limit_per_minute,
    )


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


def _football_api_player_stats_schema():
    _, ArrayType, IntegerType, StringType, StructField, StructType = _require_pyspark()
    from pyspark.sql.types import BooleanType

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
                    StructField("photo", StringType()),
                ])),
                StructField("statistics", ArrayType(StructType([
                    StructField("games", StructType([
                        StructField("minutes", IntegerType()),
                        StructField("number", IntegerType()),
                        StructField("position", StringType()),
                        StructField("rating", StringType()),
                        StructField("captain", BooleanType()),
                        StructField("substitute", BooleanType()),
                    ])),
                    StructField("offsides", IntegerType()),
                    StructField("shots", StructType([
                        StructField("total", IntegerType()),
                        StructField("on", IntegerType()),
                    ])),
                    StructField("goals", StructType([
                        StructField("total", IntegerType()),
                        StructField("conceded", IntegerType()),
                        StructField("assists", IntegerType()),
                        StructField("saves", IntegerType()),
                    ])),
                    StructField("passes", StructType([
                        StructField("total", IntegerType()),
                        StructField("key", IntegerType()),
                        StructField("accuracy", StringType()),
                    ])),
                    StructField("tackles", StructType([
                        StructField("total", IntegerType()),
                        StructField("blocks", IntegerType()),
                        StructField("interceptions", IntegerType()),
                    ])),
                    StructField("duels", StructType([
                        StructField("total", IntegerType()),
                        StructField("won", IntegerType()),
                    ])),
                    StructField("dribbles", StructType([
                        StructField("attempts", IntegerType()),
                        StructField("success", IntegerType()),
                        StructField("past", IntegerType()),
                    ])),
                    StructField("fouls", StructType([
                        StructField("drawn", IntegerType()),
                        StructField("committed", IntegerType()),
                    ])),
                    StructField("cards", StructType([
                        StructField("yellow", IntegerType()),
                        StructField("red", IntegerType()),
                    ])),
                    StructField("penalty", StructType([
                        StructField("won", IntegerType()),
                        StructField("commited", IntegerType()),
                        StructField("scored", IntegerType()),
                        StructField("missed", IntegerType()),
                        StructField("saved", IntegerType()),
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
            StructField("grid", StringType()),
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


def _football_api_fixture_statistics_schema():
    _, ArrayType, IntegerType, StringType, StructField, StructType = _require_pyspark()
    return StructType([
        StructField("response", ArrayType(StructType([
            StructField("team", StructType([
                StructField("id", IntegerType()),
                StructField("name", StringType()),
            ])),
            StructField("statistics", ArrayType(StructType([
                StructField("type", StringType()),
                # Mixed provider types (integers, "32%", nulls) land as strings; Silver casts them.
                StructField("value", StringType()),
            ]))),
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
    write_delta_target(bronze_df, bronze_path, mode=mode)


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
    envelopes = (
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
    )
    write_delta_target(envelopes, bronze_path, mode=mode)


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


def ingest_fixture_metadata_to_bronze(
    spark,
    fixture_id: int,
    *,
    api_key: Optional[str] = None,
    run_id: str = "manual",
    target_date: Optional[str] = None,
    bronze_path: str = BRONZE_FIXTURES_RAW_PATH,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
    logger: Optional[logging.Logger] = None,
) -> Mapping:
    """Fetches one `/fixtures?id=...` envelope and lands it in Bronze."""
    fixture_id = int(fixture_id)
    request_params = {"id": fixture_id}
    _log_ingestion_event(
        logger,
        logging.INFO,
        "fixture_metadata_started",
        run_id=run_id,
        stage="bronze_ingest",
        target_date=target_date,
        endpoint=FIXTURES_ENDPOINT,
        fixture_id=fixture_id,
        status=CHECKPOINT_PENDING,
    )
    upsert_endpoint_checkpoint(
        spark,
        run_id=run_id,
        target_date=target_date,
        fixture_id=fixture_id,
        endpoint=FIXTURES_ENDPOINT,
        status=CHECKPOINT_PENDING,
        checkpoint_table=checkpoint_table,
    )
    try:
        payload = _fetch_payload_with_optional_logger(
            FIXTURES_ENDPOINT,
            request_params,
            api_key=api_key,
            logger=logger,
        )
        write_bronze_raw_envelopes(
            spark,
            [payload],
            run_id=run_id,
            source_endpoint=FIXTURES_ENDPOINT,
            request_params=request_params,
            target_date=target_date,
            fixture_id=fixture_id,
            bronze_path=bronze_path,
        )
        upsert_endpoint_checkpoint(
            spark,
            run_id=run_id,
            target_date=target_date,
            fixture_id=fixture_id,
            endpoint=FIXTURES_ENDPOINT,
            status=CHECKPOINT_COMPLETED,
            response_hash=payload_hash(payload),
            checkpoint_table=checkpoint_table,
        )
        _log_ingestion_event(
            logger,
            logging.INFO,
            "fixture_metadata_completed",
            run_id=run_id,
            stage="bronze_ingest",
            target_date=target_date,
            endpoint=FIXTURES_ENDPOINT,
            fixture_id=fixture_id,
            status=CHECKPOINT_COMPLETED,
            discovered_fixtures=len(payload.get("response", [])),
        )
        return payload
    except Exception as error:
        upsert_endpoint_checkpoint(
            spark,
            run_id=run_id,
            target_date=target_date,
            fixture_id=fixture_id,
            endpoint=FIXTURES_ENDPOINT,
            status=CHECKPOINT_FAILED,
            last_error=str(error),
            checkpoint_table=checkpoint_table,
        )
        _log_ingestion_event(
            logger,
            logging.ERROR,
            "fixture_metadata_failed",
            run_id=run_id,
            stage="bronze_ingest",
            target_date=target_date,
            endpoint=FIXTURES_ENDPOINT,
            fixture_id=fixture_id,
            status=CHECKPOINT_FAILED,
            error=_safe_log_error(error),
        )
        raise


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
    eligibility = (
        spark.createDataFrame(
            rows,
            (
                "run_id string, target_date string, fixture_id int, eligibility_status string, "
                "league_id int, league_name string, league_season int, raw_fixture string, response_hash string"
            ),
        )
        .withColumn("target_date", F.to_date("target_date"))
        .withColumn("ingested_at_utc", F.current_timestamp())
    )
    write_delta_target(eligibility, bronze_path, mode="append")


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


def write_fixture_statistics_bronze(
    spark,
    api_payloads: Iterable[Mapping],
    *,
    fixture_id: int,
    run_id: str,
    bronze_path: str = BRONZE_FIXTURE_STATISTICS_RAW_PATH,
) -> None:
    write_bronze_raw_envelopes(
        spark,
        api_payloads,
        run_id=run_id,
        source_endpoint=STATISTICS_ENDPOINT,
        request_params={"fixture": int(fixture_id)},
        fixture_id=int(fixture_id),
        bronze_path=bronze_path,
    )


def fetch_football_api_payload(
    endpoint: str,
    params: Mapping,
    *,
    api_key: Optional[str] = None,
    base_url: str = BASE_URL,
    logger: Optional[logging.Logger] = None,
) -> Mapping:
    """Fetches one complete Football-API response envelope for Bronze landing."""
    client = FootballApiClient(
        api_key=api_key,
        base_url=base_url,
        request_get=requests.get,
        logger=logger,
    )
    try:
        return client.get(endpoint, params)
    except SharedFootballApiQuotaError as error:
        raise FootballApiQuotaError(str(error)) from error
    except FootballApiPayloadError as error:
        raise RuntimeError(str(error).replace("API-Sports", "Football-API")) from error


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
    logger: Optional[logging.Logger] = None,
) -> FixtureDiscoveryResult:
    """Calls `/fixtures` once for a date, lands Bronze, and filters eligibility."""
    request_params = {"date": match_date, "timezone": "UTC"}
    _log_ingestion_event(
        logger,
        logging.INFO,
        "fixture_discovery_started",
        run_id=run_id,
        stage="fixture_discovery",
        target_date=match_date,
        endpoint=FIXTURES_ENDPOINT,
        status=CHECKPOINT_PENDING,
    )
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
        payload = _fetch_payload_with_optional_logger(
            FIXTURES_ENDPOINT,
            request_params,
            api_key=api_key,
            logger=logger,
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
        _log_ingestion_event(
            logger,
            logging.INFO,
            "fixture_discovery_completed",
            run_id=run_id,
            stage="fixture_discovery",
            target_date=match_date,
            endpoint=FIXTURES_ENDPOINT,
            status=CHECKPOINT_COMPLETED,
            discovered_fixtures=len(payload.get("response", [])),
            eligible_fixtures=len(eligible),
            skipped_fixtures=len(skipped),
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
        _log_ingestion_event(
            logger,
            logging.WARNING,
            "fixture_endpoint_failed",
            run_id=run_id,
            stage="fixture_discovery",
            target_date=match_date,
            endpoint=FIXTURES_ENDPOINT,
            status=CHECKPOINT_FAILED,
            error=_safe_log_error(error),
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
    logger: Optional[logging.Logger] = None,
    endpoint_max_workers: int = 1,
    api_rate_limit_per_minute: Optional[int] = None,
) -> BronzeIngestionSummary:
    """Fetches `/fixtures/players` for explicit fixture IDs and lands Bronze rows."""
    ingested = []
    failed = 0
    skipped = 0
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
    skipped += len(plan.skipped_fixture_ids)
    _log_ingestion_event(
        logger,
        logging.INFO,
        "endpoint_plan_created",
        run_id=run_id,
        stage="bronze_ingest",
        target_date=target_date,
        endpoint=PLAYER_STATS_ENDPOINT,
        total=len(fixture_id_list),
        skipped_fixtures=len(plan.skipped_fixture_ids),
        fixtures_to_fetch=len(plan.fixture_ids_to_fetch),
    )
    for index, fixture_id in enumerate(plan.skipped_fixture_ids, start=1):
        _log_ingestion_event(
            logger,
            logging.INFO,
            "fixture_endpoint_skipped",
            run_id=run_id,
            stage="bronze_ingest",
            target_date=target_date,
            endpoint=PLAYER_STATS_ENDPOINT,
            fixture_id=fixture_id,
            index=index,
            total=len(plan.skipped_fixture_ids),
            status=CHECKPOINT_SKIPPED,
        )
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
    ingested_by_index = {}
    for result in _iter_fixture_endpoint_fetch_results(
        spark,
        plan.fixture_ids_to_fetch,
        endpoint=PLAYER_STATS_ENDPOINT,
        api_key=api_key,
        run_id=run_id,
        target_date=target_date,
        checkpoint_table=checkpoint_table,
        logger=logger,
        endpoint_max_workers=endpoint_max_workers,
        api_rate_limit_per_minute=api_rate_limit_per_minute,
    ):
        fixture_id = result.fixture_id
        index = result.index
        try:
            if result.error is not None:
                raise result.error
            payload = result.payload or {}
            has_match_data = player_stats_payload_has_match_data(payload)
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
                    status=CHECKPOINT_COMPLETED if has_match_data else CHECKPOINT_SKIPPED,
                    response_hash=payload_hash(payload),
                    checkpoint_table=checkpoint_table,
                )
            if not has_match_data:
                skipped += 1
                _log_ingestion_event(
                    logger,
                    logging.INFO,
                    "fixture_endpoint_skipped",
                    run_id=run_id,
                    stage="bronze_ingest",
                    target_date=target_date,
                    endpoint=PLAYER_STATS_ENDPOINT,
                    fixture_id=fixture_id,
                    index=index,
                    total=len(plan.fixture_ids_to_fetch),
                    status=CHECKPOINT_SKIPPED,
                )
                continue
            _log_ingestion_event(
                logger,
                logging.INFO,
                "fixture_endpoint_completed",
                run_id=run_id,
                stage="bronze_ingest",
                target_date=target_date,
                endpoint=PLAYER_STATS_ENDPOINT,
                fixture_id=fixture_id,
                index=index,
                total=len(plan.fixture_ids_to_fetch),
                status=CHECKPOINT_COMPLETED,
            )
            ingested_by_index[index] = fixture_id
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
            _log_ingestion_event(
                logger,
                logging.WARNING,
                "fixture_endpoint_failed",
                run_id=run_id,
                stage="bronze_ingest",
                target_date=target_date,
                endpoint=PLAYER_STATS_ENDPOINT,
                fixture_id=fixture_id,
                index=index,
                total=len(plan.fixture_ids_to_fetch),
                status=CHECKPOINT_FAILED,
                error=_safe_log_error(error),
            )
            failed += 1
    ingested = [fixture_id for _, fixture_id in sorted(ingested_by_index.items())]
    return BronzeIngestionSummary(
        requested_dates=(),
        discovered_fixtures=len(fixture_id_list),
        ingested_fixtures=len(ingested),
        skipped_fixtures=skipped,
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
    logger: Optional[logging.Logger] = None,
    endpoint_max_workers: int = 1,
    api_rate_limit_per_minute: Optional[int] = None,
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
    _log_ingestion_event(
        logger,
        logging.INFO,
        "endpoint_plan_created",
        run_id=run_id,
        stage="bronze_ingest",
        target_date=target_date,
        endpoint=LINEUPS_ENDPOINT,
        total=len(fixture_id_list),
        skipped_fixtures=len(plan.skipped_fixture_ids),
        fixtures_to_fetch=len(plan.fixture_ids_to_fetch),
    )
    ingested = []
    skipped = list(plan.skipped_fixture_ids)
    failed = 0
    for index, fixture_id in enumerate(plan.skipped_fixture_ids, start=1):
        _log_ingestion_event(
            logger,
            logging.INFO,
            "fixture_endpoint_skipped",
            run_id=run_id,
            stage="bronze_ingest",
            target_date=target_date,
            endpoint=LINEUPS_ENDPOINT,
            fixture_id=fixture_id,
            index=index,
            total=len(plan.skipped_fixture_ids),
            status=CHECKPOINT_SKIPPED,
        )
    ingested_by_index = {}
    for result in _iter_fixture_endpoint_fetch_results(
        spark,
        plan.fixture_ids_to_fetch,
        endpoint=LINEUPS_ENDPOINT,
        api_key=api_key,
        run_id=run_id,
        target_date=target_date,
        checkpoint_table=checkpoint_table,
        logger=logger,
        endpoint_max_workers=endpoint_max_workers,
        api_rate_limit_per_minute=api_rate_limit_per_minute,
    ):
        fixture_id = result.fixture_id
        index = result.index
        try:
            if result.error is not None:
                raise result.error
            payload = result.payload or {}
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
                _log_ingestion_event(
                    logger,
                    logging.INFO,
                    "fixture_endpoint_skipped",
                    run_id=run_id,
                    stage="bronze_ingest",
                    target_date=target_date,
                    endpoint=LINEUPS_ENDPOINT,
                    fixture_id=fixture_id,
                    index=index,
                    total=len(plan.fixture_ids_to_fetch),
                    status=CHECKPOINT_SKIPPED,
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
            _log_ingestion_event(
                logger,
                logging.INFO,
                "fixture_endpoint_completed",
                run_id=run_id,
                stage="bronze_ingest",
                target_date=target_date,
                endpoint=LINEUPS_ENDPOINT,
                fixture_id=fixture_id,
                index=index,
                total=len(plan.fixture_ids_to_fetch),
                status=CHECKPOINT_COMPLETED,
            )
            ingested_by_index[index] = fixture_id
        except Exception as error:
            failed += 1
            status = CHECKPOINT_FAILED if fixture_id in required else CHECKPOINT_SKIPPED
            _log_ingestion_event(
                logger,
                logging.WARNING,
                "fixture_endpoint_failed",
                run_id=run_id,
                stage="bronze_ingest",
                target_date=target_date,
                endpoint=LINEUPS_ENDPOINT,
                fixture_id=fixture_id,
                index=index,
                total=len(plan.fixture_ids_to_fetch),
                status=status,
                error=_safe_log_error(error),
            )
            if _supports_spark_sql(spark):
                upsert_endpoint_checkpoint(
                    spark,
                    run_id=run_id,
                    target_date=target_date,
                    fixture_id=fixture_id,
                    endpoint=LINEUPS_ENDPOINT,
                    status=status,
                    last_error=str(error),
                    checkpoint_table=checkpoint_table,
                )
            if fixture_id in required:
                raise
            skipped.append(fixture_id)
    ingested = [fixture_id for _, fixture_id in sorted(ingested_by_index.items())]
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


def ingest_fixture_statistics_for_fixtures_to_bronze(
    spark,
    fixture_ids: Iterable[int],
    *,
    api_key: Optional[str] = None,
    bronze_path: str = BRONZE_FIXTURE_STATISTICS_RAW_PATH,
    run_id: str = "manual",
    target_date: Optional[str] = None,
    completed_fixture_ids: Optional[Iterable[int]] = None,
    force_refresh: bool = False,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
    logger: Optional[logging.Logger] = None,
    endpoint_max_workers: int = 1,
    api_rate_limit_per_minute: Optional[int] = None,
) -> BronzeIngestionSummary:
    """Fetches `/fixtures/statistics` for explicit fixture IDs and lands Bronze rows."""
    failed = 0
    skipped = 0
    fixture_id_list = [int(fixture_id) for fixture_id in fixture_ids]
    completed_ids = set(completed_fixture_ids or ())
    if completed_fixture_ids is None and _supports_spark_sql(spark):
        completed_ids = completed_checkpoint_fixture_ids(
            spark,
            endpoint=STATISTICS_ENDPOINT,
            target_date=target_date,
            checkpoint_table=checkpoint_table,
        )
    plan = endpoint_ingestion_plan(
        fixture_id_list,
        completed_fixture_ids=completed_ids,
        force_refresh=force_refresh,
    )
    skipped += len(plan.skipped_fixture_ids)
    _log_ingestion_event(
        logger,
        logging.INFO,
        "endpoint_plan_created",
        run_id=run_id,
        stage="bronze_ingest",
        target_date=target_date,
        endpoint=STATISTICS_ENDPOINT,
        total=len(fixture_id_list),
        skipped_fixtures=len(plan.skipped_fixture_ids),
        fixtures_to_fetch=len(plan.fixture_ids_to_fetch),
    )
    for index, fixture_id in enumerate(plan.skipped_fixture_ids, start=1):
        _log_ingestion_event(
            logger,
            logging.INFO,
            "fixture_endpoint_skipped",
            run_id=run_id,
            stage="bronze_ingest",
            target_date=target_date,
            endpoint=STATISTICS_ENDPOINT,
            fixture_id=fixture_id,
            index=index,
            total=len(plan.skipped_fixture_ids),
            status=CHECKPOINT_SKIPPED,
        )
    ingested_by_index = {}
    for result in _iter_fixture_endpoint_fetch_results(
        spark,
        plan.fixture_ids_to_fetch,
        endpoint=STATISTICS_ENDPOINT,
        api_key=api_key,
        run_id=run_id,
        target_date=target_date,
        checkpoint_table=checkpoint_table,
        logger=logger,
        endpoint_max_workers=endpoint_max_workers,
        api_rate_limit_per_minute=api_rate_limit_per_minute,
    ):
        fixture_id = result.fixture_id
        index = result.index
        try:
            if result.error is not None:
                raise result.error
            payload = result.payload or {}
            has_data = fixture_statistics_payload_has_data(payload)
            if _supports_spark_sql(spark):
                write_fixture_statistics_bronze(
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
                    endpoint=STATISTICS_ENDPOINT,
                    status=CHECKPOINT_COMPLETED if has_data else CHECKPOINT_SKIPPED,
                    response_hash=payload_hash(payload),
                    checkpoint_table=checkpoint_table,
                )
            if not has_data:
                skipped += 1
                _log_ingestion_event(
                    logger,
                    logging.INFO,
                    "fixture_endpoint_skipped",
                    run_id=run_id,
                    stage="bronze_ingest",
                    target_date=target_date,
                    endpoint=STATISTICS_ENDPOINT,
                    fixture_id=fixture_id,
                    index=index,
                    total=len(plan.fixture_ids_to_fetch),
                    status=CHECKPOINT_SKIPPED,
                )
                continue
            _log_ingestion_event(
                logger,
                logging.INFO,
                "fixture_endpoint_completed",
                run_id=run_id,
                stage="bronze_ingest",
                target_date=target_date,
                endpoint=STATISTICS_ENDPOINT,
                fixture_id=fixture_id,
                index=index,
                total=len(plan.fixture_ids_to_fetch),
                status=CHECKPOINT_COMPLETED,
            )
            ingested_by_index[index] = fixture_id
        except Exception as error:
            if _supports_spark_sql(spark):
                upsert_endpoint_checkpoint(
                    spark,
                    run_id=run_id,
                    target_date=target_date,
                    fixture_id=fixture_id,
                    endpoint=STATISTICS_ENDPOINT,
                    status=CHECKPOINT_FAILED,
                    last_error=str(error),
                    checkpoint_table=checkpoint_table,
                )
            _log_ingestion_event(
                logger,
                logging.WARNING,
                "fixture_endpoint_failed",
                run_id=run_id,
                stage="bronze_ingest",
                target_date=target_date,
                endpoint=STATISTICS_ENDPOINT,
                fixture_id=fixture_id,
                index=index,
                total=len(plan.fixture_ids_to_fetch),
                status=CHECKPOINT_FAILED,
                error=_safe_log_error(error),
            )
            failed += 1
    ingested = [fixture_id for _, fixture_id in sorted(ingested_by_index.items())]
    return BronzeIngestionSummary(
        requested_dates=(),
        discovered_fixtures=len(fixture_id_list),
        ingested_fixtures=len(ingested),
        skipped_fixtures=skipped,
        failed_fixtures=failed,
        fixture_ids=tuple(ingested),
        eligible_fixtures=len(fixture_id_list),
        statistics_ingested=len(ingested),
        statistics_skipped=skipped,
    )


def ingest_senior_mens_international_bronze(
    spark,
    *,
    api_key: Optional[str] = None,
    run_id: Optional[str] = None,
    target_date: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    completed_only: bool = False,
    force_refresh: bool = False,
    include_lineups: bool = True,
    include_statistics: bool = True,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    required_fixture_ids: Optional[Iterable[int]] = None,
    bronze_fixtures_path: str = BRONZE_FIXTURES_RAW_PATH,
    bronze_eligibility_path: str = BRONZE_FIXTURES_ELIGIBILITY_PATH,
    bronze_player_stats_path: str = BRONZE_FOOTBALL_MATCH_RAW_PATH,
    bronze_lineups_path: str = BRONZE_LINEUPS_RAW_PATH,
    bronze_statistics_path: str = BRONZE_FIXTURE_STATISTICS_RAW_PATH,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
    logger: Optional[logging.Logger] = None,
    endpoint_max_workers: int = 1,
    api_rate_limit_per_minute: Optional[int] = None,
) -> BronzeIngestionSummary:
    """Runs Bronze fixture discovery plus eligible player-stat/lineup ingestion."""
    date_selection = resolve_ingestion_dates(
        spark,
        target_date=target_date,
        date_from=date_from,
        date_to=date_to,
        checkpoint_table=checkpoint_table,
        lookahead_days=lookahead_days,
        lookback_days=lookback_days,
    )
    dates = list(date_selection.dates)

    active_run_id = run_id or f"intl-{utc_now_iso()}"
    date_scope_fields = _date_scope_log_fields(dates)
    _log_ingestion_event(
        logger,
        logging.INFO,
        "bronze_ingest_started",
        run_id=active_run_id,
        stage="bronze_ingest",
        total=len(dates),
        status=CHECKPOINT_PENDING,
        date_selection_mode=date_selection.mode,
        **date_scope_fields,
    )
    all_fixture_ids = []
    discovered_count = 0
    skipped_fixture_count = 0
    player_ingested = 0
    lineups_ingested = 0
    lineups_skipped = 0
    statistics_ingested = 0
    statistics_skipped = 0
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
            logger=logger,
        )
        discovered_count += len(discovery.raw_payload.get("response", []))
        skipped_fixture_count += len(discovery.skipped_fixtures)
        fixture_ids = discovery.eligible_fixture_ids
        player_stat_fixture_ids = fixture_ids_for_player_stats(discovery.eligible_fixtures)
        lineup_fixture_ids = fixture_ids_for_lineups(discovery.eligible_fixtures)
        statistics_fixture_ids = fixture_ids_for_fixture_statistics(discovery.eligible_fixtures)
        all_fixture_ids.extend(fixture_ids)

        player_summary = ingest_player_stats_for_fixtures_to_bronze(
            spark,
            player_stat_fixture_ids,
            api_key=api_key,
            run_id=active_run_id,
            target_date=match_date,
            force_refresh=force_refresh,
            bronze_path=bronze_player_stats_path,
            checkpoint_table=checkpoint_table,
            logger=logger,
            endpoint_max_workers=endpoint_max_workers,
            api_rate_limit_per_minute=api_rate_limit_per_minute,
        )
        player_ingested += player_summary.player_stat_payloads_ingested
        skipped_fixture_count += player_summary.skipped_fixtures
        failures += player_summary.failed_fixtures

        if include_lineups:
            lineup_summary = ingest_lineups_for_fixtures_to_bronze(
                spark,
                lineup_fixture_ids,
                api_key=api_key,
                run_id=active_run_id,
                target_date=match_date,
                force_refresh=force_refresh,
                required_fixture_ids=required_fixture_ids,
                bronze_path=bronze_lineups_path,
                checkpoint_table=checkpoint_table,
                logger=logger,
                endpoint_max_workers=endpoint_max_workers,
                api_rate_limit_per_minute=api_rate_limit_per_minute,
            )
            lineups_ingested += lineup_summary.lineups_ingested
            lineups_skipped += lineup_summary.lineups_skipped
            failures += lineup_summary.failed_fixtures

        if include_statistics:
            statistics_summary = ingest_fixture_statistics_for_fixtures_to_bronze(
                spark,
                statistics_fixture_ids,
                api_key=api_key,
                run_id=active_run_id,
                target_date=match_date,
                force_refresh=force_refresh,
                bronze_path=bronze_statistics_path,
                checkpoint_table=checkpoint_table,
                logger=logger,
                endpoint_max_workers=endpoint_max_workers,
                api_rate_limit_per_minute=api_rate_limit_per_minute,
            )
            statistics_ingested += statistics_summary.statistics_ingested
            statistics_skipped += statistics_summary.statistics_skipped
            failures += statistics_summary.failed_fixtures

    summary = BronzeIngestionSummary(
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
        statistics_ingested=statistics_ingested,
        statistics_skipped=statistics_skipped,
    )
    _log_ingestion_event(
        logger,
        logging.INFO,
        "bronze_ingest_completed",
        run_id=active_run_id,
        stage="bronze_ingest",
        total=len(dates),
        status=CHECKPOINT_COMPLETED,
        date_selection_mode=date_selection.mode,
        discovered_fixtures=summary.discovered_fixtures,
        eligible_fixtures=summary.eligible_fixtures,
        skipped_fixtures=summary.skipped_fixtures,
        failed_fixtures=summary.failed_fixtures,
        **date_scope_fields,
    )
    return summary


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
    upsert_endpoint_checkpoints(
        spark,
        [{
            "run_id": run_id,
            "target_date": target_date,
            "fixture_id": fixture_id,
            "endpoint": endpoint,
            "status": status,
            "attempt_count": attempt_count,
            "last_error": last_error,
            "response_hash": response_hash,
        }],
        checkpoint_table=checkpoint_table,
    )


def upsert_endpoint_checkpoints(
    spark,
    checkpoint_rows: Iterable[Mapping],
    *,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
) -> None:
    normalized_rows = []
    for row in checkpoint_rows:
        normalized_rows.append((
            row["run_id"],
            row.get("target_date"),
            int(row["fixture_id"]) if row.get("fixture_id") is not None else None,
            row["endpoint"],
            row["status"],
            int(row.get("attempt_count", 1)),
            row.get("last_error"),
            row.get("response_hash"),
        ))
    if not normalized_rows:
        return
    ensure_fixture_endpoint_checkpoint_table(spark, checkpoint_table=checkpoint_table)
    F, *_ = _require_pyspark()
    staged = (
        spark.createDataFrame(
            normalized_rows,
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
    try:
        filtered = checkpoint.where(
            (F.col("endpoint") == endpoint)
            & (F.col("status") == CHECKPOINT_COMPLETED)
            & F.col("fixture_id").isNotNull()
        )
        if target_date:
            filtered = filtered.where(F.col("target_date") == F.to_date(F.lit(target_date)))
        return {int(row.fixture_id) for row in filtered.select("fixture_id").distinct().collect()}
    except Exception:
        return set()


def latest_completed_checkpoint_date(
    spark,
    *,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
    endpoints: Sequence[str] = (FIXTURES_ENDPOINT, PLAYER_STATS_ENDPOINT, LINEUPS_ENDPOINT),
    max_target_date: Optional[str] = None,
) -> Optional[str]:
    """Returns the latest completed target date recorded by endpoint checkpoints."""
    if spark is None or not hasattr(spark, "table"):
        return None
    F, *_ = _require_pyspark()
    try:
        checkpoint = spark.table(checkpoint_table)
    except Exception:
        return None

    try:
        filtered = checkpoint.where(
            (F.col("status") == CHECKPOINT_COMPLETED)
            & F.col("target_date").isNotNull()
            & F.col("endpoint").isin([str(endpoint) for endpoint in endpoints])
        )
        if max_target_date:
            filtered = filtered.where(F.col("target_date") <= F.to_date(F.lit(max_target_date)))
        rows = filtered.agg(F.max("target_date").alias("target_date")).collect()
    except Exception:
        return None
    if not rows:
        return None
    value = rows[0]["target_date"] if hasattr(rows[0], "__getitem__") else rows[0].target_date
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


