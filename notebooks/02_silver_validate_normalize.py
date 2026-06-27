# Databricks notebook source
from football_analytics.databricks_ingestion import transform_bronze_to_silver

silver_df = transform_bronze_to_silver(spark)
display(silver_df.limit(20))

