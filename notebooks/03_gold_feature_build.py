# Databricks notebook source
from football_analytics.databricks_ingestion import (
    build_gold_player_shot_features,
    build_gold_rating_baseline,
    build_gold_team_match_context,
    transform_silver_to_gold_sapm,
)

team_context_df = build_gold_team_match_context(spark)
rating_baseline_df = build_gold_rating_baseline(spark)
player_features_df = build_gold_player_shot_features(spark)
gold_df = transform_silver_to_gold_sapm(spark)

display({
    "team_context": team_context_df.count(),
    "rating_baseline": rating_baseline_df.count(),
    "player_shot_features": player_features_df.count(),
    "sapm": gold_df.count(),
})

