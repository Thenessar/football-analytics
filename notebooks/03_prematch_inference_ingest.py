# Databricks notebook source
"""Pre-match inference steps 1-3 (business_logic.md §14.3).

Discovers fixtures kicking off inside the polling window (or uses an explicit
fixture_id), refreshes fixture metadata, and force-refreshes lineups into
Bronze so the dbt task that follows can rebuild the lineup-dependent Silver
and Gold models.
"""
import logging
from datetime import datetime, timedelta, timezone

from football_analytics.databricks.config import DatabricksPipelineConfig, load_config_from_env
from football_analytics.databricks.logging import configure_json_logging
from football_analytics.databricks.tables import table_name
from football_analytics.databricks_ingestion import (
    DEFAULT_API_RATE_LIMIT_PER_MINUTE,
    ingest_fixture_metadata_to_bronze,
    ingest_lineups_for_fixtures_to_bronze,
    utc_now_iso,
)

dbutils.widgets.text("fixture_id", "")
dbutils.widgets.text("window_minutes", "75")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("catalog", "football_analytics")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("gold_schema", "gold")
dbutils.widgets.text("api_rate_limit_per_minute", str(DEFAULT_API_RATE_LIMIT_PER_MINUTE))

env_config = load_config_from_env()
config = DatabricksPipelineConfig(
    catalog=dbutils.widgets.get("catalog"),
    bronze_schema=dbutils.widgets.get("bronze_schema"),
    silver_schema=dbutils.widgets.get("silver_schema"),
    gold_schema=dbutils.widgets.get("gold_schema"),
    api_key=env_config.api_key,
)
fixture_id_widget = dbutils.widgets.get("fixture_id").strip()
window_minutes = max(1, int(dbutils.widgets.get("window_minutes").strip() or "75"))
run_id = dbutils.widgets.get("run_id").strip() or f"inference-{utc_now_iso()}"
api_rate_limit_per_minute = int(dbutils.widgets.get("api_rate_limit_per_minute").strip() or DEFAULT_API_RATE_LIMIT_PER_MINUTE)
api_key = config.api_key or dbutils.secrets.get("football-api", "api-football-key")
logger = configure_json_logging(level=logging.INFO, logger_name="football_analytics.prematch_inference_ingest")

if fixture_id_widget:
    fixture_ids = [int(fixture_id_widget)]
else:
    now_utc = datetime.now(timezone.utc)
    window_end = now_utc + timedelta(minutes=window_minutes)
    fixtures_table = table_name(config, "silver", "stg_football__fixtures")
    upcoming = spark.sql(
        f"""
        SELECT fixture_id
        FROM {fixtures_table}
        WHERE status_short = 'NS'
          AND fixture_date_utc >= TIMESTAMP'{now_utc.isoformat()}'
          AND fixture_date_utc <= TIMESTAMP'{window_end.isoformat()}'
        """
    ).collect()
    fixture_ids = sorted(int(row.fixture_id) for row in upcoming)

summaries = []
for fixture_id in fixture_ids:
    ingest_fixture_metadata_to_bronze(
        spark,
        fixture_id,
        api_key=api_key,
        run_id=run_id,
        bronze_path=table_name(config, "bronze", "football_fixtures_raw"),
        checkpoint_table=table_name(config, "bronze", "ingestion_state_checkpoint"),
        logger=logger,
    )
    # Lineups keep changing until confirmed, so always refetch inside the
    # trigger window regardless of checkpoint state.
    lineup_summary = ingest_lineups_for_fixtures_to_bronze(
        spark,
        [fixture_id],
        api_key=api_key,
        bronze_path=table_name(config, "bronze", "football_lineups_raw"),
        run_id=run_id,
        force_refresh=True,
        checkpoint_table=table_name(config, "bronze", "ingestion_state_checkpoint"),
        logger=logger,
        api_rate_limit_per_minute=api_rate_limit_per_minute,
    )
    summaries.append({
        "fixture_id": fixture_id,
        "lineups_ingested": lineup_summary.lineups_ingested,
        "lineups_skipped": lineup_summary.lineups_skipped,
    })

dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.jobs.taskValues.set(key="fixture_ids", value=",".join(str(f) for f in fixture_ids))
display({
    "run_id": run_id,
    "window_minutes": window_minutes,
    "fixture_ids": fixture_ids,
    "lineup_summaries": summaries,
})
