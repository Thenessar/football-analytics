# Databricks notebook source
from football_analytics.databricks.config import DatabricksPipelineConfig
from football_analytics.databricks.tables import table_name
from football_analytics.inference import PREDICTION_TABLE_NAME, ensure_player_event_predictions_table

dbutils.widgets.text("fixture_id", "")
dbutils.widgets.text("run_id", "manual")
dbutils.widgets.text("target_date", "")
dbutils.widgets.text("date_from", "")
dbutils.widgets.text("date_to", "")
dbutils.widgets.text("lookahead_days", "7")
dbutils.widgets.text("catalog", "football_analytics")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("gold_schema", "gold")
dbutils.widgets.dropdown("force_refresh", "false", ["false", "true"])
dbutils.widgets.dropdown("include_lineups", "true", ["true", "false"])

run_id = dbutils.widgets.get("run_id")
fixture_id = dbutils.widgets.get("fixture_id")
target_date = dbutils.widgets.get("target_date")
date_from = dbutils.widgets.get("date_from")
date_to = dbutils.widgets.get("date_to")
lookahead_days = dbutils.widgets.get("lookahead_days")
config = DatabricksPipelineConfig(
    catalog=dbutils.widgets.get("catalog"),
    bronze_schema=dbutils.widgets.get("bronze_schema"),
    silver_schema=dbutils.widgets.get("silver_schema"),
    gold_schema=dbutils.widgets.get("gold_schema"),
)

for schema_name in {
    config.bronze_schema,
    config.silver_schema,
    config.gold_schema,
}:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.catalog}.{schema_name}")

# The prediction table is written by the inference job but tested by dbt as a
# source; create it up front so dbt source tests never hit a missing table.
ensure_player_event_predictions_table(
    spark,
    table_name(config, "gold", PREDICTION_TABLE_NAME),
)

dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.jobs.taskValues.set(key="fixture_id", value=fixture_id)
dbutils.jobs.taskValues.set(key="target_date", value=target_date)
dbutils.jobs.taskValues.set(key="date_from", value=date_from)
dbutils.jobs.taskValues.set(key="date_to", value=date_to)
dbutils.jobs.taskValues.set(key="lookahead_days", value=lookahead_days)
dbutils.jobs.taskValues.set(key="force_refresh", value=dbutils.widgets.get("force_refresh"))
dbutils.jobs.taskValues.set(key="include_lineups", value=dbutils.widgets.get("include_lineups"))
