# Databricks notebook source
from football_analytics.databricks.config import DatabricksPipelineConfig, load_config_from_env
from football_analytics.databricks.tables import table_name
from football_analytics.databricks_ingestion import (
    ingest_fixture_player_stats_to_delta,
    ingest_senior_mens_international_bronze,
    transform_bronze_to_silver,
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
run_id = dbutils.widgets.get("run_id").strip() or None
force_refresh = dbutils.widgets.get("force_refresh").strip().lower() == "true"
include_lineups = dbutils.widgets.get("include_lineups").strip().lower() == "true"
api_key = config.api_key or dbutils.secrets.get("football-api", "api-football-key")

if fixture_id:
    silver_df = ingest_fixture_player_stats_to_delta(
        spark,
        int(fixture_id),
        api_key=api_key,
        bronze_path=table_name(config, "bronze", "football_match_raw"),
        silver_path=table_name(config, "silver", "football_player_match_stats"),
    )
    display({"mode": "fixture_id", "fixture_id": int(fixture_id)})
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
    )
    display(summary.as_dict())
    silver_df = transform_bronze_to_silver(
        spark,
        bronze_path=table_name(config, "bronze", "football_match_raw"),
        silver_path=table_name(config, "silver", "football_player_match_stats"),
    )

display(silver_df.limit(20))
