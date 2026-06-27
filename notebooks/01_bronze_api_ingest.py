# Databricks notebook source
from football_analytics.databricks.config import load_config_from_env
from football_analytics.databricks_ingestion import (
    ingest_fixture_player_stats_to_delta,
    ingest_world_cup_player_stats_bronze,
    transform_bronze_to_silver,
)

dbutils.widgets.text("fixture_id", "")
dbutils.widgets.text("target_date", "")
dbutils.widgets.text("date_from", "")
dbutils.widgets.text("date_to", "")

config = load_config_from_env()
fixture_id = dbutils.widgets.get("fixture_id").strip()
target_date = dbutils.widgets.get("target_date").strip()
date_from = dbutils.widgets.get("date_from").strip()
date_to = dbutils.widgets.get("date_to").strip()
api_key = config.api_key or dbutils.secrets.get("football-api", "api-football-key")

if fixture_id:
    silver_df = ingest_fixture_player_stats_to_delta(
        spark,
        int(fixture_id),
        api_key=api_key,
    )
    display({"mode": "fixture_id", "fixture_id": int(fixture_id)})
else:
    summary = ingest_world_cup_player_stats_bronze(
        spark,
        api_key=api_key,
        target_date=target_date or None,
        date_from=date_from or None,
        date_to=date_to or None,
    )
    display(summary.as_dict())
    silver_df = transform_bronze_to_silver(spark)

display(silver_df.limit(20))
