# Databricks notebook source
from football_analytics.databricks_ingestion import write_fifa_rankings_seed_table

dbutils.widgets.text("fixture_id", "1489437")
dbutils.widgets.text("run_id", "manual")
dbutils.widgets.text("target_date", "")
dbutils.widgets.text("date_from", "")
dbutils.widgets.text("date_to", "")
dbutils.widgets.dropdown("force_refresh", "false", ["false", "true"])
dbutils.widgets.dropdown("include_lineups", "true", ["true", "false"])
dbutils.widgets.dropdown("load_rankings_seed", "true", ["true", "false"])

run_id = dbutils.widgets.get("run_id")
fixture_id = dbutils.widgets.get("fixture_id")
target_date = dbutils.widgets.get("target_date")
date_from = dbutils.widgets.get("date_from")
date_to = dbutils.widgets.get("date_to")

dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.jobs.taskValues.set(key="fixture_id", value=fixture_id)
dbutils.jobs.taskValues.set(key="target_date", value=target_date)
dbutils.jobs.taskValues.set(key="date_from", value=date_from)
dbutils.jobs.taskValues.set(key="date_to", value=date_to)
dbutils.jobs.taskValues.set(key="force_refresh", value=dbutils.widgets.get("force_refresh"))
dbutils.jobs.taskValues.set(key="include_lineups", value=dbutils.widgets.get("include_lineups"))

if dbutils.widgets.get("load_rankings_seed").lower() == "true":
    rankings_seed_df = write_fifa_rankings_seed_table(spark)
    display(rankings_seed_df.limit(20))
