# Databricks notebook source
from football_analytics.databricks.config import load_config_from_env
from football_analytics.databricks_ingestion import ingest_fixture_player_stats_to_delta

dbutils.widgets.text("fixture_id", "1489437")

config = load_config_from_env()
fixture_id = int(dbutils.widgets.get("fixture_id"))

silver_df = ingest_fixture_player_stats_to_delta(
    spark,
    fixture_id,
    api_key=config.api_key or dbutils.secrets.get("football-api", "api-football-key"),
)
display(silver_df.limit(20))
