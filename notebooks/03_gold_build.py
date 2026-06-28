# Databricks notebook source
from football_analytics.databricks.config import DatabricksPipelineConfig
from football_analytics.databricks.tables import table_name
from football_analytics.databricks_ingestion import (
    build_gold_player_shot_features,
    build_gold_rating_baseline,
    build_gold_team_match_context,
    transform_silver_to_gold_sapm,
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

team_context_df = build_gold_team_match_context(
    spark,
    silver_fixtures_path=table_name(config, "silver", "football_fixtures"),
    gold_path=table_name(config, "gold", "football_team_match_context"),
)
rating_baseline_df = build_gold_rating_baseline(
    spark,
    seed_table=table_name(config, "bronze", "fifa_mens_world_ranking_seed"),
    gold_path=table_name(config, "gold", "football_rating_baseline"),
)
player_features_df = build_gold_player_shot_features(
    spark,
    silver_player_stats_path=table_name(config, "silver", "football_player_match_stats"),
    silver_fixtures_path=table_name(config, "silver", "football_fixtures"),
    gold_path=table_name(config, "gold", "football_player_shot_features"),
)
gold_df = transform_silver_to_gold_sapm(
    spark,
    silver_path=table_name(config, "silver", "football_player_match_stats"),
    fixture_context_path=table_name(config, "gold", "football_team_match_context"),
    gold_path=table_name(config, "gold", "football_player_sapm"),
)

display({
    "team_context": team_context_df.count(),
    "rating_baseline": rating_baseline_df.count(),
    "player_shot_features": player_features_df.count(),
    "sapm": gold_df.count(),
})

