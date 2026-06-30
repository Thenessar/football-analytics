# Databricks notebook source
import logging

from football_analytics.databricks.config import DatabricksPipelineConfig, load_config_from_env
from football_analytics.databricks.logging import configure_json_logging
from football_analytics.databricks.tables import table_name
from football_analytics.databricks_ingestion import (
    DEFAULT_API_RATE_LIMIT_PER_MINUTE,
    ingest_fixture_metadata_to_bronze,
    ingest_lineups_for_fixtures_to_bronze,
    ingest_player_stats_for_fixtures_to_bronze,
    ingest_senior_mens_international_bronze,
    utc_now_iso,
)

dbutils.widgets.text("fixture_id", "")
dbutils.widgets.text("target_date", "")
dbutils.widgets.text("date_from", "")
dbutils.widgets.text("date_to", "")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("catalog", "football_analytics")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("gold_schema", "gold")
dbutils.widgets.text("ops_schema", "ops")
dbutils.widgets.dropdown("force_refresh", "false", ["false", "true"])
dbutils.widgets.dropdown("include_lineups", "true", ["true", "false"])
dbutils.widgets.text("endpoint_max_workers", "8")
dbutils.widgets.text("api_rate_limit_per_minute", str(DEFAULT_API_RATE_LIMIT_PER_MINUTE))


def positive_int_widget(name: str, default: int, *, maximum: int | None = None) -> int:
    try:
        value = int((dbutils.widgets.get(name) or "").strip() or default)
    except (TypeError, ValueError):
        value = default
    value = max(1, value)
    if maximum is not None:
        value = min(maximum, value)
    return value

env_config = load_config_from_env()
config = DatabricksPipelineConfig(
    catalog=dbutils.widgets.get("catalog"),
    bronze_schema=dbutils.widgets.get("bronze_schema"),
    silver_schema=dbutils.widgets.get("silver_schema"),
    gold_schema=dbutils.widgets.get("gold_schema"),
    ops_schema=dbutils.widgets.get("ops_schema"),
    api_key=env_config.api_key,
)
fixture_id = dbutils.widgets.get("fixture_id").strip()
target_date = dbutils.widgets.get("target_date").strip()
date_from = dbutils.widgets.get("date_from").strip()
date_to = dbutils.widgets.get("date_to").strip()
run_id = dbutils.widgets.get("run_id").strip() or f"intl-{utc_now_iso()}"
force_refresh = dbutils.widgets.get("force_refresh").strip().lower() == "true"
include_lineups = dbutils.widgets.get("include_lineups").strip().lower() == "true"
endpoint_max_workers = positive_int_widget("endpoint_max_workers", 8)
api_rate_limit_per_minute = positive_int_widget(
    "api_rate_limit_per_minute",
    DEFAULT_API_RATE_LIMIT_PER_MINUTE,
    maximum=300,
)
api_key = config.api_key or dbutils.secrets.get("football-api", "api-football-key")
logger = configure_json_logging(level=logging.INFO, logger_name="football_analytics.bronze_ingest")
logger.info(
    "bronze_parallelism_configured",
    extra={
        "event": "bronze_parallelism_configured",
        "run_id": run_id,
        "stage": "bronze_ingest",
        "endpoint_max_workers": endpoint_max_workers,
        "api_rate_limit_per_minute": api_rate_limit_per_minute,
    },
)

if fixture_id:
    fixture_payload = ingest_fixture_metadata_to_bronze(
        spark,
        int(fixture_id),
        api_key=api_key,
        run_id=run_id,
        target_date=target_date or None,
        bronze_path=table_name(config, "bronze", "football_fixtures_raw"),
        checkpoint_table=table_name(config, "ops", "ingestion_state_checkpoint"),
        logger=logger,
    )
    player_summary = ingest_player_stats_for_fixtures_to_bronze(
        spark,
        [int(fixture_id)],
        api_key=api_key,
        bronze_path=table_name(config, "bronze", "football_match_raw"),
        run_id=run_id,
        target_date=target_date or None,
        force_refresh=force_refresh,
        checkpoint_table=table_name(config, "ops", "ingestion_state_checkpoint"),
        logger=logger,
        endpoint_max_workers=endpoint_max_workers,
        api_rate_limit_per_minute=api_rate_limit_per_minute,
    )
    lineup_summary = None
    if include_lineups:
        lineup_summary = ingest_lineups_for_fixtures_to_bronze(
            spark,
            [int(fixture_id)],
            api_key=api_key,
            bronze_path=table_name(config, "bronze", "football_lineups_raw"),
            run_id=run_id,
            target_date=target_date or None,
            force_refresh=force_refresh,
            checkpoint_table=table_name(config, "ops", "ingestion_state_checkpoint"),
            logger=logger,
            endpoint_max_workers=endpoint_max_workers,
            api_rate_limit_per_minute=api_rate_limit_per_minute,
        )
    display({
        "mode": "fixture_id",
        "fixture_id": int(fixture_id),
        "fixture_rows": len(fixture_payload.get("response", [])),
        "player_stats": player_summary.as_dict(),
        "lineups": lineup_summary.as_dict() if lineup_summary else None,
    })
else:
    summary = ingest_senior_mens_international_bronze(
        spark,
        api_key=api_key,
        run_id=run_id,
        target_date=target_date or None,
        date_from=date_from or None,
        date_to=date_to or None,
        force_refresh=force_refresh,
        include_lineups=include_lineups,
        bronze_fixtures_path=table_name(config, "bronze", "football_fixtures_raw"),
        bronze_eligibility_path=table_name(config, "bronze", "football_fixture_eligibility"),
        bronze_player_stats_path=table_name(config, "bronze", "football_match_raw"),
        bronze_lineups_path=table_name(config, "bronze", "football_lineups_raw"),
        checkpoint_table=table_name(config, "ops", "ingestion_state_checkpoint"),
        logger=logger,
        endpoint_max_workers=endpoint_max_workers,
        api_rate_limit_per_minute=api_rate_limit_per_minute,
    )
    display(summary.as_dict())
