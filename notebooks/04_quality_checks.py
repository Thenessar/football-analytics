# Databricks notebook source
from football_analytics.databricks.config import DatabricksPipelineConfig
from football_analytics.databricks.ops_views import create_bronze_ops_views
from football_analytics.databricks.tables import table_name
from football_analytics.databricks_ingestion import natural_key_merge_plans

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

plans = natural_key_merge_plans()
display([(name, plan.table, ", ".join(plan.keys), plan.predicate) for name, plan in plans.items()])

ops_views = create_bronze_ops_views(
    spark,
    config,
    checkpoint_table=table_name(config, "ops", "ingestion_state_checkpoint"),
)
display(ops_views.as_dict())
display(spark.table(ops_views.run_summary))
