# Databricks notebook source
dbutils.widgets.text("fixture_id", "1489437")
dbutils.widgets.text("run_id", "manual")

run_id = dbutils.widgets.get("run_id")
fixture_id = dbutils.widgets.get("fixture_id")

dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.jobs.taskValues.set(key="fixture_id", value=fixture_id)
