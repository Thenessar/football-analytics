import json
from typing import Iterable, Mapping, Optional

import requests

from football_analytics.config import BASE_URL, HEADERS

BRONZE_FOOTBALL_MATCH_RAW_PATH = "/mnt/syndicate/bronze/football_match_raw"
SILVER_PLAYER_MATCH_STATS_PATH = "/mnt/syndicate/silver/football_player_match_stats"

ACCENTED_CHARS = (
    "ÀÁÂÃÄÅĀĂĄÇĆČÐĎÈÉÊËĒĔĖĘĚÌÍÎÏĪĮİŁÑŃŇÒÓÔÕÖØŌŐŘŚŞŠÙÚÛÜŪŮŰÝŸŽŹŻ"
    "àáâãäåāăąçćčðďèéêëēĕėęěìíîïīįıłñńňòóôõöøōőřśşšùúûüūůűýÿžźż"
)
ASCII_CHARS = (
    "AAAAAAAAACCCDDEEEEEEEEEIIIIIIILNNNOOOOOOOORSSSUUUUUUUYYYZZZ"
    "aaaaaaaaacccddeeeeeeeeeiiiiiilnnnoooooooorsssuuuuuuuyyzzz"
)


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
        )
        for payload in api_payloads
    ]
    if not rows:
        return

    bronze_df = (
        spark.createDataFrame(rows, "fixture_id int, source_endpoint string, raw_payload string")
        .withColumn("ingested_at_utc", F.current_timestamp())
    )
    bronze_df.write.format("delta").mode(mode).save(bronze_path)


def fetch_football_api_payload(
    endpoint: str,
    params: Mapping,
    *,
    api_key: Optional[str] = None,
    base_url: str = BASE_URL,
) -> Mapping:
    """Fetches one complete Football-API response envelope for Bronze landing."""
    headers = HEADERS.copy()
    if api_key:
        headers["x-rapidapi-key"] = api_key
        headers["x-apisports-key"] = api_key

    response = requests.get(
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        headers=headers,
        params=dict(params),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(f"Football-API returned errors for {endpoint}: {payload['errors']}")
    return payload


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
