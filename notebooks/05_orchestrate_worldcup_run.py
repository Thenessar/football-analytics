# Databricks notebook source
run_id = dbutils.jobs.taskValues.get(taskKey="config", key="run_id", default=dbutils.widgets.get("run_id"))
fixture_id = dbutils.widgets.get("fixture_id")

dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.jobs.taskValues.set(key="fixture_id", value=fixture_id)

