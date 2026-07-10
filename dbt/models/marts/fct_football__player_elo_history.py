import pandas as pd
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


# fixture_id is nullable because the builder also emits one current-state row
# per (team, player) after the chronological pass (is_current_state = true,
# fixture_id null). Those rows carry the post-history modifiers that inference
# rows join by player for future fixtures (FA-101).
PLAYER_ELO_SCHEMA = StructType([
    StructField("fixture_id", IntegerType(), True),
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
    StructField("is_current_state", BooleanType(), False),
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
    frame = history[columns].astype(object)
    frame = frame.where(pd.notna(frame), None)
    # Nullable integer columns arrive as float64 (pandas has no nullable int
    # in a mixed frame); Spark's IntegerType rejects Python floats.
    for field in PLAYER_ELO_SCHEMA.fields:
        if isinstance(field.dataType, IntegerType):
            frame[field.name] = [
                None if value is None else int(value)
                for value in frame[field.name]
            ]
    return session.createDataFrame(frame, PLAYER_ELO_SCHEMA)
