# Databricks notebook source
# MAGIC %md
# MAGIC # World Cup 2026 Databricks Ingestion
# MAGIC
# MAGIC Creates the Unity Catalog layout and populates the Delta tables consumed by
# MAGIC `notebooks/data_discovery.ipynb`.
# MAGIC
# MAGIC Tables created:
# MAGIC
# MAGIC - `football_analytics.bronze.raw_fixture_payloads`
# MAGIC - `football_analytics.silver.matches`
# MAGIC - `football_analytics.silver.player_match_stats`
# MAGIC - `football_analytics.gold.team_elo_matrix`
# MAGIC - `football_analytics.gold.match_context_weights`
# MAGIC - `football_analytics.gold.player_shooting_decay_features`
# MAGIC
# MAGIC The preferred path is a fresh API pull from API-Football. A local JSON bootstrap
# MAGIC path is included for cases where you upload the existing cache to the raw volume.

# COMMAND ----------

# MAGIC %python

from datetime import date, datetime, timedelta, timezone
import json
import time
from typing import Any, Dict, Iterable, List, Optional

import requests
from pyspark.sql import functions as F
from pyspark.sql import types as T

dbutils.widgets.text("catalog", "football_analytics")
dbutils.widgets.text("league_ids", "")
dbutils.widgets.text("season_start", "2022")
dbutils.widgets.text("season_end", "2026")
dbutils.widgets.text("date_from", "2022-11-07")
dbutils.widgets.text("date_to", "2026-07-19")
dbutils.widgets.text("date_chunk_days", "7")
dbutils.widgets.text("recency_half_life_days", "365")
dbutils.widgets.text("timezone", "UTC")
dbutils.widgets.text("api_host", "v3.football.api-sports.io")
dbutils.widgets.text("secret_scope", "football-api")
dbutils.widgets.text("secret_key", "api-football-key")
dbutils.widgets.text("api_key_widget_fallback", "")
dbutils.widgets.dropdown("source_mode", "api", ["api", "uploaded_json"])
dbutils.widgets.text("uploaded_json_path", "")
dbutils.widgets.text("request_sleep_seconds", "0.75")

CATALOG = dbutils.widgets.get("catalog")
LEAGUE_IDS = [
    int(value.strip())
    for value in dbutils.widgets.get("league_ids").split(",")
    if value.strip()
]
SEASON_START = int(dbutils.widgets.get("season_start"))
SEASON_END = int(dbutils.widgets.get("season_end"))
DATE_FROM = date.fromisoformat(dbutils.widgets.get("date_from"))
DATE_TO = date.fromisoformat(dbutils.widgets.get("date_to"))
DATE_CHUNK_DAYS = int(dbutils.widgets.get("date_chunk_days"))
RECENCY_HALF_LIFE_DAYS = float(dbutils.widgets.get("recency_half_life_days"))
API_HOST = dbutils.widgets.get("api_host")
TIMEZONE = dbutils.widgets.get("timezone")
SOURCE_MODE = dbutils.widgets.get("source_mode")
UPLOADED_JSON_PATH = dbutils.widgets.get("uploaded_json_path").strip()
REQUEST_SLEEP_SECONDS = float(dbutils.widgets.get("request_sleep_seconds"))

BRONZE = f"{CATALOG}.bronze"
SILVER = f"{CATALOG}.silver"
GOLD = f"{CATALOG}.gold"
RAW_VOLUME = f"/Volumes/{CATALOG}/bronze/raw_api"
RAW_JSON_DIR = f"{RAW_VOLUME}/fixtures"

COMPLETED_STATUS = {"FT", "AET", "PEN"}
HIGH_STAKES_KEYWORDS = [
    "world cup",
    "euro",
    "copa america",
    "copa américa",
    "africa cup",
    "asian cup",
    "gold cup",
    "nations league",
    "qualifier",
    "qualification",
    "playoff",
    "play-off",
]

def sql_ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Unsafe SQL identifier: {name}")
    return f"`{name}`"

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Databricks Runtime 13.3 LTS+ with Unity Catalog is expected for Volumes.

# COMMAND ----------

# MAGIC %python

spark.sql(f"CREATE CATALOG IF NOT EXISTS {sql_ident(CATALOG)} COMMENT 'Football analytics lakehouse'")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {BRONZE} COMMENT 'Raw API payloads and landing objects'")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER} COMMENT 'Cleaned match and player-level tables'")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD} COMMENT 'Discovery-ready analytical feature tables'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {BRONZE}.raw_api COMMENT 'Raw API-Football payload archive'")
dbutils.fs.mkdirs(RAW_JSON_DIR)

display(spark.sql(f"SHOW VOLUMES IN {BRONZE}"))

# COMMAND ----------

# MAGIC %python

def get_api_key() -> str:
    try:
        scope = dbutils.widgets.get("secret_scope")
        key = dbutils.widgets.get("secret_key")
        if scope and key:
            value = dbutils.secrets.get(scope=scope, key=key)
            if value:
                return value
    except Exception:
        pass
    fallback = dbutils.widgets.get("api_key_widget_fallback")
    if fallback:
        return fallback
    raise ValueError(
        "API key not found. Create a Databricks secret or set api_key_widget_fallback for a one-off run."
    )

def api_get(path: str, params: Dict[str, Any], api_key: str) -> List[Dict[str, Any]]:
    url = f"https://{API_HOST}{path}"
    headers = {
        "x-apisports-key": api_key,
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": API_HOST,
    }
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(f"API-Football error from {path}: {payload['errors']}")
    return payload.get("response", [])

def date_range(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)

def date_chunks(start: date, end: date, chunk_days: int) -> Iterable[tuple[date, date]]:
    if chunk_days < 1:
        raise ValueError("date_chunk_days must be >= 1")
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)

def existing_bronze_fixture_ids() -> set[int]:
    if not spark.catalog.tableExists(f"{BRONZE}.raw_fixture_payloads"):
        return set()
    return {
        int(row.fixture_id)
        for row in spark.table(f"{BRONZE}.raw_fixture_payloads")
        .select("fixture_id")
        .distinct()
        .collect()
    }

def is_high_stakes_league(league_payload: Dict[str, Any]) -> int:
    league_name = str((league_payload or {}).get("name") or "").casefold()
    if "friendly" in league_name or "friendlies" in league_name:
        return 0
    return int(any(keyword in league_name for keyword in HIGH_STAKES_KEYWORDS))

def is_international_fixture(row: Dict[str, Any]) -> bool:
    teams = row.get("teams") or {}
    home_national = (teams.get("home") or {}).get("national")
    away_national = (teams.get("away") or {}).get("national")
    if home_national is True and away_national is True:
        return True
    league_name = str(((row.get("league") or {}).get("name") or "")).casefold()
    return (
        "friendly" in league_name
        or "friendlies" in league_name
        or any(keyword in league_name for keyword in HIGH_STAKES_KEYWORDS)
    )

def fetch_completed_fixtures(api_key: str) -> List[Dict[str, Any]]:
    fixtures: Dict[int, Dict[str, Any]] = {}
    seen_fixture_ids = existing_bronze_fixture_ids()
    for chunk_start, chunk_end in date_chunks(DATE_FROM, DATE_TO, DATE_CHUNK_DAYS):
        print(f"Fetching completed international fixtures for {chunk_start} through {chunk_end}")
        for match_date in date_range(chunk_start, chunk_end):
            for league_id in (LEAGUE_IDS or [None]):
                params = {
                    "date": match_date.isoformat(),
                    "timezone": TIMEZONE,
                }
                if league_id is not None:
                    params["league"] = league_id
                    params["season"] = next(
                        season
                        for season in range(SEASON_START, SEASON_END + 1)
                        if season == match_date.year
                    )
                rows = api_get("/fixtures", params, api_key)
                for row in rows:
                    league_type = ((row.get("league") or {}).get("type") or "").casefold()
                    status = ((row.get("fixture") or {}).get("status") or {}).get("short")
                    fixture_id = (row.get("fixture") or {}).get("id")
                    if (
                        fixture_id
                        and int(fixture_id) not in seen_fixture_ids
                        and status in COMPLETED_STATUS
                        and (not league_type or league_type == "cup")
                        and (league_id is not None or is_international_fixture(row))
                    ):
                        fixtures[int(fixture_id)] = row
                time.sleep(REQUEST_SLEEP_SECONDS)
    return list(fixtures.values())

def fetch_fixture_payloads() -> List[Dict[str, Any]]:
    api_key = get_api_key()
    fixture_rows = fetch_completed_fixtures(api_key)
    records = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for fixture_row in fixture_rows:
        fixture_id = int(fixture_row["fixture"]["id"])
        player_statistics = api_get("/fixtures/players", {"fixture": fixture_id}, api_key)
        records.append(
            {
                "fixture_id": fixture_id,
                "fetched_at": fetched_at,
                "league_id": int((fixture_row.get("league") or {}).get("id") or 0),
                "season": int((fixture_row.get("league") or {}).get("season") or fixture_row["fixture"]["date"][:4]),
                "match_date": fixture_row["fixture"]["date"][:10],
                "is_high_stakes": is_high_stakes_league(fixture_row.get("league") or {}),
                "fixture": fixture_row.get("fixture"),
                "league": fixture_row.get("league"),
                "teams": fixture_row.get("teams"),
                "goals": fixture_row.get("goals"),
                "score": fixture_row.get("score"),
                "player_statistics": player_statistics,
            }
        )
        time.sleep(REQUEST_SLEEP_SECONDS)
    return records

def records_from_uploaded_cache(path: str) -> List[Dict[str, Any]]:
    if not path:
        raise ValueError("uploaded_json_path is required when source_mode=uploaded_json")
    text = "\n".join(row.value for row in spark.read.text(path).collect())
    cache = json.loads(text)
    records = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for raw_fixture_id, payload in cache.items():
        match_info = payload.get("match_info", {})
        home_name = match_info.get("home_team")
        away_name = match_info.get("away_team")
        team_ids = {
            ((team_payload.get("team") or {}).get("name")): ((team_payload.get("team") or {}).get("id"))
            for team_payload in payload.get("player_statistics", [])
        }
        records.append(
            {
                "fixture_id": int(raw_fixture_id),
                "fetched_at": fetched_at,
                "league_id": int(match_info.get("league_id") or 0),
                "season": int(str(match_info.get("date") or DATE_FROM.isoformat())[:4]),
                "match_date": str(match_info.get("date") or DATE_FROM.isoformat())[:10],
                "is_high_stakes": is_high_stakes_league(match_info.get("league") or {}),
                "fixture": {
                    "id": int(raw_fixture_id),
                    "date": match_info.get("date"),
                    "referee": match_info.get("referee"),
                    "venue": {"name": match_info.get("venue")},
                    "status": {"short": "FT"},
                },
                "league": match_info.get("league") or {"id": int(match_info.get("league_id") or 0), "season": int(str(match_info.get("date") or DATE_FROM.isoformat())[:4])},
                "teams": {
                    "home": {"id": team_ids.get(home_name), "name": home_name},
                    "away": {"id": team_ids.get(away_name), "name": away_name},
                },
                "goals": match_info.get("score", {}),
                "score": {"fulltime": match_info.get("score", {})},
                "player_statistics": payload.get("player_statistics", []),
            }
        )
    return records

records = (
    fetch_fixture_payloads()
    if SOURCE_MODE == "api"
    else records_from_uploaded_cache(UPLOADED_JSON_PATH)
)

if not records and not spark.catalog.tableExists(f"{BRONZE}.raw_fixture_payloads"):
    raise RuntimeError("No completed fixture payloads were fetched. Check dates, league, season, and API access.")
elif not records:
    print("No new fixture payloads fetched; using existing bronze checkpoint table.")
else:
    landing_path = f"{RAW_JSON_DIR}/run_date={datetime.now(timezone.utc).date().isoformat()}"
    dbutils.fs.rm(landing_path, recurse=True)
    landing_rows = [
        (
            int(record["fixture_id"]),
            str(record["fetched_at"]),
            int(record["league_id"]),
            int(record["season"]),
            str(record["match_date"]),
            int(record["is_high_stakes"]),
            json.dumps(record["fixture"], ensure_ascii=False),
            json.dumps(record["league"], ensure_ascii=False),
            json.dumps(record["teams"], ensure_ascii=False),
            json.dumps(record["goals"], ensure_ascii=False),
            json.dumps(record["score"], ensure_ascii=False),
            json.dumps(record["player_statistics"], ensure_ascii=False),
        )
        for record in records
    ]
    landing_schema = T.StructType(
        [
            T.StructField("fixture_id", T.LongType(), False),
            T.StructField("fetched_at", T.StringType(), False),
            T.StructField("league_id", T.IntegerType(), False),
            T.StructField("season", T.IntegerType(), False),
            T.StructField("match_date", T.StringType(), False),
            T.StructField("is_high_stakes", T.IntegerType(), False),
            T.StructField("fixture_json", T.StringType(), True),
            T.StructField("league_json", T.StringType(), True),
            T.StructField("teams_json", T.StringType(), True),
            T.StructField("goals_json", T.StringType(), True),
            T.StructField("score_json", T.StringType(), True),
            T.StructField("player_statistics_json", T.StringType(), True),
        ]
    )
    landing_df = spark.createDataFrame(landing_rows, landing_schema)
    json_schema_sample = records[0]
    raw_df = (
        landing_df
        .withColumn("fixture", F.from_json("fixture_json", F.schema_of_json(F.lit(json.dumps(json_schema_sample["fixture"], ensure_ascii=False)))))
        .withColumn("league", F.from_json("league_json", F.schema_of_json(F.lit(json.dumps(json_schema_sample["league"], ensure_ascii=False)))))
        .withColumn("teams", F.from_json("teams_json", F.schema_of_json(F.lit(json.dumps(json_schema_sample["teams"], ensure_ascii=False)))))
        .withColumn("goals", F.from_json("goals_json", F.schema_of_json(F.lit(json.dumps(json_schema_sample["goals"], ensure_ascii=False)))))
        .withColumn("score", F.from_json("score_json", F.schema_of_json(F.lit(json.dumps(json_schema_sample["score"], ensure_ascii=False)))))
        .withColumn("player_statistics", F.from_json("player_statistics_json", F.schema_of_json(F.lit(json.dumps(json_schema_sample["player_statistics"], ensure_ascii=False)))))
        .drop("fixture_json", "league_json", "teams_json", "goals_json", "score_json", "player_statistics_json")
    )
    raw_df.write.mode("overwrite").json(landing_path)

    bronze_df = spark.read.json(landing_path)
    bronze_df.createOrReplaceTempView("new_raw_fixture_payloads")
    if not spark.catalog.tableExists(f"{BRONZE}.raw_fixture_payloads"):
        (
            bronze_df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .partitionBy("match_date")
            .saveAsTable(f"{BRONZE}.raw_fixture_payloads")
        )
    else:
        spark.sql(f"""
            MERGE INTO {BRONZE}.raw_fixture_payloads AS target
            USING new_raw_fixture_payloads AS source
            ON target.fixture_id = source.fixture_id
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)

display(spark.table(f"{BRONZE}.raw_fixture_payloads").select("fixture_id", "match_date", "fetched_at", "season").orderBy("match_date", "fixture_id"))

# COMMAND ----------

# MAGIC %python

bronze = spark.table(f"{BRONZE}.raw_fixture_payloads")

matches = (
    bronze.select(
        F.col("fixture_id").cast("long").alias("fixture_id"),
        F.to_timestamp("fixture.date").alias("match_date"),
        F.col("fixture.referee").alias("referee"),
        F.col("fixture.venue.name").alias("venue_name"),
        F.col("league_id").cast("int").alias("league_id"),
        F.col("league.name").alias("league_name"),
        F.col("season").cast("int").alias("season"),
        F.col("is_high_stakes").cast("int").alias("is_high_stakes"),
        F.col("teams.home.id").cast("long").alias("home_team_id"),
        F.col("teams.home.name").alias("home_team_name"),
        F.col("teams.away.id").cast("long").alias("away_team_id"),
        F.col("teams.away.name").alias("away_team_name"),
        F.col("goals.home").cast("int").alias("score_fulltime_home"),
        F.col("goals.away").cast("int").alias("score_fulltime_away"),
        F.col("fixture.status.short").alias("status_short"),
    )
    .dropDuplicates(["fixture_id"])
)

team_stats = (
    bronze.select(
        "fixture_id",
        F.to_timestamp("fixture.date").alias("match_date"),
        F.col("teams.home.id").cast("long").alias("home_team_id"),
        F.col("teams.home.name").alias("home_team_name"),
        F.col("teams.away.id").cast("long").alias("away_team_id"),
        F.col("teams.away.name").alias("away_team_name"),
        F.explode_outer("player_statistics").alias("team_payload"),
    )
)

players_exploded = (
    team_stats
    .select(
        "fixture_id",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        F.col("team_payload.team.id").cast("long").alias("team_id"),
        F.col("team_payload.team.name").alias("team_name"),
        F.explode_outer("team_payload.players").alias("player_payload"),
    )
)

player_match_stats = (
    players_exploded
    .select(
        "fixture_id",
        "match_date",
        "team_id",
        "team_name",
        F.when(F.col("team_id") == F.col("home_team_id"), F.col("away_team_id")).otherwise(F.col("home_team_id")).alias("opponent_team_id"),
        F.when(F.col("team_id") == F.col("home_team_id"), F.col("away_team_name")).otherwise(F.col("home_team_name")).alias("opponent_team_name"),
        F.col("player_payload.player.id").cast("long").alias("player_id"),
        F.col("player_payload.player.name").alias("player_name"),
        F.col("player_payload.player.photo").alias("player_photo"),
        F.explode_outer("player_payload.statistics").alias("stats"),
    )
    .select(
        "fixture_id",
        "team_id",
        "team_name",
        "opponent_team_id",
        "opponent_team_name",
        "player_id",
        "player_name",
        "player_photo",
        F.col("stats.games.position").alias("position"),
        F.col("stats.games.minutes").cast("double").alias("minutes"),
        F.col("stats.games.rating").cast("double").alias("rating"),
        F.col("stats.games.substitute").cast("boolean").alias("substitute"),
        F.coalesce(F.col("stats.shots.total").cast("double"), F.lit(0.0)).alias("shots_total"),
        F.coalesce(F.col("stats.shots.on").cast("double"), F.lit(0.0)).alias("shots_on"),
        F.coalesce(F.col("stats.goals.total").cast("double"), F.lit(0.0)).alias("goals_total"),
        F.coalesce(F.col("stats.goals.assists").cast("double"), F.lit(0.0)).alias("assists"),
        F.coalesce(F.col("stats.passes.key").cast("double"), F.lit(0.0)).alias("key_passes"),
        F.coalesce(F.col("stats.passes.key").cast("double"), F.lit(0.0)).alias("chances_created"),
        F.coalesce(F.col("stats.passes.total").cast("double"), F.lit(0.0)).alias("passes_total"),
        F.coalesce(F.col("stats.tackles.total").cast("double"), F.lit(0.0)).alias("tackles_total"),
        F.coalesce(F.col("stats.tackles.interceptions").cast("double"), F.lit(0.0)).alias("interceptions"),
        F.coalesce(F.col("stats.duels.won").cast("double"), F.lit(0.0)).alias("duels_won"),
        F.coalesce(F.col("stats.cards.yellow").cast("double"), F.lit(0.0)).alias("yellow_cards"),
        F.coalesce(F.col("stats.cards.red").cast("double"), F.lit(0.0)).alias("red_cards"),
    )
    .where(F.col("player_id").isNotNull())
)

(
    matches.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SILVER}.matches")
)
(
    player_match_stats.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SILVER}.player_match_stats")
)

display(matches.orderBy("match_date"))
display(player_match_stats.orderBy("fixture_id", "team_name", "player_name").limit(50))

# COMMAND ----------

# MAGIC %python

team_match = (
    spark.table(f"{SILVER}.player_match_stats")
    .groupBy("fixture_id", "team_id", "team_name", "opponent_team_id", "opponent_team_name")
    .agg(
        F.sum("shots_total").alias("team_shots"),
        F.sum("goals_total").alias("team_player_goals"),
        F.sum("key_passes").alias("team_key_passes"),
        F.sum("minutes").alias("team_player_minutes"),
    )
)

team_match_with_result = (
    team_match.alias("tm")
    .join(spark.table(f"{SILVER}.matches").alias("m"), "fixture_id", "left")
    .withColumn(
        "team_goals",
        F.when(F.col("tm.team_id") == F.col("m.home_team_id"), F.col("m.score_fulltime_home")).otherwise(F.col("m.score_fulltime_away")),
    )
    .withColumn(
        "opponent_goals",
        F.when(F.col("tm.team_id") == F.col("m.home_team_id"), F.col("m.score_fulltime_away")).otherwise(F.col("m.score_fulltime_home")),
    )
    .withColumn("goal_diff", F.coalesce(F.col("team_goals"), F.lit(0)) - F.coalesce(F.col("opponent_goals"), F.lit(0)))
    .withColumn("result_points", F.when(F.col("goal_diff") > 0, 3.0).when(F.col("goal_diff") == 0, 1.0).otherwise(0.0))
)

def elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

elo_rows = []
ratings: Dict[int, float] = {}
ordered_matches = (
    spark.table(f"{SILVER}.matches")
    .where(F.col("home_team_id").isNotNull() & F.col("away_team_id").isNotNull())
    .orderBy("match_date", "fixture_id")
    .collect()
)
for match in ordered_matches:
    home_id = int(match.home_team_id)
    away_id = int(match.away_team_id)
    home_pre = ratings.get(home_id, 1500.0)
    away_pre = ratings.get(away_id, 1500.0)
    home_goals = match.score_fulltime_home or 0
    away_goals = match.score_fulltime_away or 0
    if home_goals > away_goals:
        home_score, away_score = 1.0, 0.0
    elif home_goals < away_goals:
        home_score, away_score = 0.0, 1.0
    else:
        home_score, away_score = 0.5, 0.5
    k_factor = 40.0 if int(match.is_high_stakes or 0) == 1 else 20.0
    home_expected = elo_expected(home_pre, away_pre)
    away_expected = elo_expected(away_pre, home_pre)
    home_post = home_pre + k_factor * (home_score - home_expected)
    away_post = away_pre + k_factor * (away_score - away_expected)
    ratings[home_id] = home_post
    ratings[away_id] = away_post
    elo_rows.extend([
        {
            "fixture_id": int(match.fixture_id),
            "match_date": match.match_date,
            "team_id": home_id,
            "team_name": match.home_team_name,
            "opponent_team_id": away_id,
            "opponent_team_name": match.away_team_name,
            "pre_team_elo": float(home_pre),
            "pre_opponent_elo": float(away_pre),
            "team_elo": float(home_post),
            "opponent_elo": float(away_post),
            "is_high_stakes": int(match.is_high_stakes or 0),
        },
        {
            "fixture_id": int(match.fixture_id),
            "match_date": match.match_date,
            "team_id": away_id,
            "team_name": match.away_team_name,
            "opponent_team_id": home_id,
            "opponent_team_name": match.home_team_name,
            "pre_team_elo": float(away_pre),
            "pre_opponent_elo": float(home_pre),
            "team_elo": float(away_post),
            "opponent_elo": float(home_post),
            "is_high_stakes": int(match.is_high_stakes or 0),
        },
    ])

team_elo_schema = T.StructType([
    T.StructField("fixture_id", T.LongType(), False),
    T.StructField("match_date", T.TimestampType(), True),
    T.StructField("team_id", T.LongType(), False),
    T.StructField("team_name", T.StringType(), True),
    T.StructField("opponent_team_id", T.LongType(), False),
    T.StructField("opponent_team_name", T.StringType(), True),
    T.StructField("pre_team_elo", T.DoubleType(), False),
    T.StructField("pre_opponent_elo", T.DoubleType(), False),
    T.StructField("team_elo", T.DoubleType(), False),
    T.StructField("opponent_elo", T.DoubleType(), False),
    T.StructField("is_high_stakes", T.IntegerType(), False),
])
team_elo_matrix = spark.createDataFrame(elo_rows, team_elo_schema)

match_context_weights = (
    spark.table(f"{SILVER}.matches")
    .withColumn(
        "competition_phase_weight",
        F.when(F.col("is_high_stakes") == 1, F.lit(1.25)).otherwise(F.lit(1.00)),
    )
    .withColumn("w_stake", F.col("competition_phase_weight"))
    .select("fixture_id", "w_stake")
    .join(
        team_elo_matrix.groupBy("fixture_id").agg(
            F.avg(F.abs(F.col("team_elo") - F.col("opponent_elo"))).alias("avg_elo_gap")
        ),
        "fixture_id",
        "left",
    )
    .withColumn("w_opp", F.least(F.lit(1.25), F.greatest(F.lit(0.85), F.lit(1.0) + F.coalesce(F.col("avg_elo_gap"), F.lit(0.0)) / F.lit(1000.0))))
    .select("fixture_id", "w_stake", "w_opp")
)

player_shooting_decay_features = (
    spark.table(f"{SILVER}.player_match_stats")
    .where(F.col("minutes").isNotNull() & (F.col("minutes") > 0))
    .withColumn(
        "observation_age_days",
        F.greatest(
            F.lit(0.0),
            (
                F.unix_timestamp(F.to_timestamp(F.lit(DATE_TO.isoformat())))
                - F.unix_timestamp("match_date")
            ) / F.lit(86400.0),
        ),
    )
    .withColumn(
        "recency_weight",
        F.exp(-F.lit(0.6931471805599453 / RECENCY_HALF_LIFE_DAYS) * F.col("observation_age_days")),
    )
    .withColumn("shots_total_decayed", F.col("shots_total") * F.col("recency_weight"))
    .withColumn("shots_on_decayed", F.col("shots_on") * F.col("recency_weight"))
    .withColumn("goals_total_decayed", F.col("goals_total") * F.col("recency_weight"))
    .withColumn("minutes_decayed", F.col("minutes") * F.col("recency_weight"))
    .groupBy("player_id", "player_name", "position", "team_id", "team_name")
    .agg(
        F.sum("shots_total_decayed").alias("shots_total_decayed"),
        F.sum("shots_on_decayed").alias("shots_on_decayed"),
        F.sum("goals_total_decayed").alias("goals_total_decayed"),
        F.sum("minutes_decayed").alias("minutes_decayed"),
        F.max("match_date").alias("latest_match_date"),
        F.countDistinct("fixture_id").alias("matches_observed"),
    )
    .withColumn(
        "shots_per_90_decayed",
        F.when(F.col("minutes_decayed") > 0, F.col("shots_total_decayed") / F.col("minutes_decayed") * F.lit(90.0)).otherwise(F.lit(0.0)),
    )
)

(
    team_elo_matrix.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD}.team_elo_matrix")
)
(
    match_context_weights.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD}.match_context_weights")
)
(
    player_shooting_decay_features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD}.player_shooting_decay_features")
)

display(team_elo_matrix.orderBy("fixture_id", "team_id"))
display(match_context_weights.orderBy("fixture_id"))
display(player_shooting_decay_features.orderBy(F.desc("shots_per_90_decayed")).limit(50))

# COMMAND ----------

# MAGIC %python

quality_checks = {
    f"{BRONZE}.raw_fixture_payloads": spark.table(f"{BRONZE}.raw_fixture_payloads").count(),
    f"{SILVER}.matches": spark.table(f"{SILVER}.matches").count(),
    f"{SILVER}.player_match_stats": spark.table(f"{SILVER}.player_match_stats").count(),
    f"{GOLD}.team_elo_matrix": spark.table(f"{GOLD}.team_elo_matrix").count(),
    f"{GOLD}.match_context_weights": spark.table(f"{GOLD}.match_context_weights").count(),
    f"{GOLD}.player_shooting_decay_features": spark.table(f"{GOLD}.player_shooting_decay_features").count(),
}

display(spark.createDataFrame([(k, v) for k, v in quality_checks.items()], ["table_name", "row_count"]))

if quality_checks[f"{SILVER}.matches"] == 0:
    raise RuntimeError("Silver matches table is empty.")
if quality_checks[f"{SILVER}.player_match_stats"] == 0:
    raise RuntimeError("Silver player_match_stats table is empty.")

print("Ingestion complete. Run notebooks/data_discovery.ipynb with the football_analytics defaults.")
