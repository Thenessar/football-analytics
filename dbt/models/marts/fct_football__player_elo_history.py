from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


PLAYER_ELO_SCHEMA = StructType([
    StructField("fixture_id", IntegerType(), False),
    StructField("fixture_date_utc", TimestampType(), True),
    StructField("team_id", IntegerType(), False),
    StructField("team_name", StringType(), True),
    StructField("player_id", IntegerType(), False),
    StructField("player_name", StringType(), True),
    StructField("games_minutes", DoubleType(), True),
    StructField("team_elo_general_pre", DoubleType(), True),
    StructField("team_elo_attack_pre", DoubleType(), True),
    StructField("team_elo_defense_pre", DoubleType(), True),
    StructField("player_offensive_modifier_pre", DoubleType(), True),
    StructField("player_defensive_modifier_pre", DoubleType(), True),
    StructField("player_offensive_elo_pre", DoubleType(), True),
    StructField("player_defensive_elo_pre", DoubleType(), True),
    StructField("player_offensive_rating_pre", DoubleType(), True),
    StructField("player_defensive_rating_pre", DoubleType(), True),
    StructField("missed_fixture_count_pre", IntegerType(), True),
    StructField("player_offensive_signal_p90", DoubleType(), True),
    StructField("player_defensive_signal_p90", DoubleType(), True),
    StructField("team_offensive_signal_p90", DoubleType(), True),
    StructField("team_defensive_signal_p90", DoubleType(), True),
])


def model(dbt, session):
    dbt.config(materialized="table")

    from football_analytics.elo import build_player_elo_history

    appearances = dbt.ref("fct_football__player_match_features").toPandas()
    team_history = dbt.ref("fct_football__team_elo_history").toPandas()
    history = build_player_elo_history(appearances, team_history)

    if history.empty:
        return session.createDataFrame([], PLAYER_ELO_SCHEMA)

    columns = [field.name for field in PLAYER_ELO_SCHEMA.fields]
    return session.createDataFrame(history[columns], PLAYER_ELO_SCHEMA)
