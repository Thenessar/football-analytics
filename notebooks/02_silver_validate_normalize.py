# Databricks notebook source
from football_analytics.databricks_ingestion import (
    transform_bronze_fixtures_to_silver,
    transform_bronze_lineups_to_silver,
    transform_bronze_to_silver,
)

fixtures_df = transform_bronze_fixtures_to_silver(spark)
silver_df = transform_bronze_to_silver(spark)
lineups_df = transform_bronze_lineups_to_silver(spark)

display({
    "fixtures": fixtures_df.count(),
    "player_stats": silver_df.count(),
    "lineups": lineups_df.count(),
})

