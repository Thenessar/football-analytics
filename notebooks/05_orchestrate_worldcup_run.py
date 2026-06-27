# Databricks notebook source
from football_analytics.databricks_ingestion import write_fifa_rankings_seed_table

dbutils.widgets.text("fixture_id", "1489437")
dbutils.widgets.text("run_id", "manual")
dbutils.widgets.dropdown("load_rankings_seed", "true", ["true", "false"])

run_id = dbutils.widgets.get("run_id")
fixture_id = dbutils.widgets.get("fixture_id")

dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.jobs.taskValues.set(key="fixture_id", value=fixture_id)

if dbutils.widgets.get("load_rankings_seed").lower() == "true":
    rankings_seed_df = write_fifa_rankings_seed_table(spark)
    display(rankings_seed_df.limit(20))
