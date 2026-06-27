# Databricks notebook source
from football_analytics.databricks_ingestion import transform_silver_to_gold_sapm

gold_df = transform_silver_to_gold_sapm(spark)
display(gold_df.limit(20))

