# Databricks notebook source
from football_analytics.databricks.config import DatabricksPipelineConfig
from football_analytics.databricks.tables import table_name
from football_analytics.databricks_ingestion import (
    transform_bronze_fixtures_to_silver,
    transform_bronze_lineups_to_silver,
    transform_bronze_to_silver,
)

dbutils.widgets.text("catalog", "football_analytics")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("gold_schema", "gold")
dbutils.widgets.text("ops_schema", "ops")

config = DatabricksPipelineConfig(
    catalog=dbutils.widgets.get("catalog"),
    bronze_schema=dbutils.widgets.get("bronze_schema"),
    silver_schema=dbutils.widgets.get("silver_schema"),
    gold_schema=dbutils.widgets.get("gold_schema"),
    ops_schema=dbutils.widgets.get("ops_schema"),
)

fixtures_df = transform_bronze_fixtures_to_silver(
    spark,
    bronze_path=table_name(config, "bronze", "football_fixtures_raw"),
    silver_path=table_name(config, "silver", "football_fixtures"),
)
silver_df = transform_bronze_to_silver(
    spark,
    bronze_path=table_name(config, "bronze", "football_match_raw"),
    silver_path=table_name(config, "silver", "football_player_match_stats"),
)
lineups_df = transform_bronze_lineups_to_silver(
    spark,
    bronze_path=table_name(config, "bronze", "football_lineups_raw"),
    silver_path=table_name(config, "silver", "football_lineups"),
)

display({
    "fixtures": fixtures_df.count(),
    "player_stats": silver_df.count(),
    "lineups": lineups_df.count(),
})

