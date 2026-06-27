import json
from datetime import date, timedelta
from typing import Iterable, Mapping, Optional

import requests

from football_analytics.config import BASE_URL, HEADERS
from football_analytics.api import FootballApiClient, is_quota_error_payload, payload_hash
from football_analytics.api.exceptions import FootballApiPayloadError
from football_analytics.api.exceptions import FootballApiQuotaError as SharedFootballApiQuotaError
from football_analytics.modeling import build_empirical_bayes_shot_rate_pandas_udf
from football_analytics.quality.validators import ValidationError, validate_world_cup_fixture
from football_analytics.storage.delta_io import build_merge_plan

BRONZE_FOOTBALL_MATCH_RAW_PATH = "/mnt/syndicate/bronze/football_match_raw"
SILVER_PLAYER_MATCH_STATS_PATH = "/mnt/syndicate/silver/football_player_match_stats"
GOLD_PLAYER_SAPM_PATH = "/mnt/syndicate/gold/football_player_sapm"
INGESTION_STATE_CHECKPOINT_TABLE = "default.ingestion_state_checkpoint"
DEAD_LETTER_TABLE = "default.football_ingestion_dead_letter"
CHECKPOINT_PENDING = "PENDING"
CHECKPOINT_COMPLETED = "COMPLETED"
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
    start_date: str = "2022-11-07",
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


def quarantine_payload(
    spark,
    *,
    payload: Mapping,
    reason: str,
    fixture_id: Optional[int] = None,
    stage: str = "silver_validate",
    dead_letter_table: str = DEAD_LETTER_TABLE,
) -> None:
    """Writes invalid payloads to a Delta dead-letter table for later inspection."""
    F, *_ = _require_pyspark()
    rows = [(
        int(fixture_id) if fixture_id is not None else None,
        stage,
        reason,
        json.dumps(payload, ensure_ascii=False),
        payload_hash(payload),
    )]
    (
        spark.createDataFrame(
            rows,
            "fixture_id int, stage string, reason string, raw_payload string, response_hash string",
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
    """Returns True for valid World Cup fixtures and quarantines rejected records."""
    try:
        validate_world_cup_fixture(fixture)
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
            ("fixture_id", "league_id", "season", "status", "response_hash", "updated_at_utc"),
        ),
        "player_stats": build_merge_plan(
            "silver.player_stats",
            ("fixture_id", "team_id", "player_id"),
            ("fixture_id", "team_id", "player_id", "minutes", "shots_total", "response_hash", "updated_at_utc"),
        ),
        "lineups": build_merge_plan(
            "silver.lineups",
            ("fixture_id", "team_id", "player_id"),
            ("fixture_id", "team_id", "player_id", "position", "response_hash", "updated_at_utc"),
        ),
    }


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


def ensure_ingestion_checkpoint_table(
    spark,
    *,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
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
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
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


def update_ingestion_checkpoint(
    spark,
    window_start_date: str,
    window_end_date: str,
    *,
    status: str,
    records_ingested: int,
    checkpoint_table: str = INGESTION_STATE_CHECKPOINT_TABLE,
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


def transform_bronze_to_silver(
    spark,
    *,
    bronze_path: str = BRONZE_FOOTBALL_MATCH_RAW_PATH,
    silver_path: str = SILVER_PLAYER_MATCH_STATS_PATH,
    mode: str = "overwrite",
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
            F.col("ingested_at_utc"),
        )
        .where(F.col("player_id").isNotNull())
        .withColumn("player_name_normalized", normalized_name_sql("player_name"))
        .withColumn("team_name_normalized", normalized_name_sql("team_name"))
        .dropDuplicates([
            "fixture_id",
            "team_id",
            "player_id",
            "games_minutes",
            "games_position",
            "shots_total",
            "shots_on",
            "goals_total",
        ])
    )

    silver.write.format("delta").mode(mode).save(silver_path)
    return silver


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
