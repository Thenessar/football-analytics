from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


TEAM_ELO_SCHEMA = StructType([
    StructField("fixture_id", IntegerType(), False),
    StructField("fixture_date_utc", TimestampType(), True),
    StructField("team_id", IntegerType(), False),
    StructField("team_name", StringType(), True),
    StructField("opponent_team_id", IntegerType(), True),
    StructField("opponent_team_name", StringType(), True),
    StructField("home_away", StringType(), True),
    StructField("team_elo_general_pre", DoubleType(), True),
    StructField("opponent_elo_general_pre", DoubleType(), True),
    StructField("team_elo_attack_pre", DoubleType(), True),
    StructField("team_elo_defense_pre", DoubleType(), True),
    StructField("opponent_elo_attack_pre", DoubleType(), True),
    StructField("opponent_elo_defense_pre", DoubleType(), True),
    StructField("expected_goals_for_pre", DoubleType(), True),
    StructField("expected_goals_against_pre", DoubleType(), True),
    StructField("team_elo_attack_post", DoubleType(), True),
    StructField("team_elo_defense_post", DoubleType(), True),
])


def model(dbt, session):
    dbt.config(materialized="table")

    from football_analytics.elo import build_team_elo_history

    fixtures = dbt.ref("stg_football__fixtures").toPandas()
    rankings = dbt.ref("dim_football__rating_baseline").toPandas()
    stats_df = dbt.ref("stg_football__player_match_stats").toPandas()

    stats_agg = stats_df.groupby(["fixture_id", "team_id"]).agg(
        shots_total=("shots_total", "sum"),
        shots_on=("shots_on", "sum"),
    ).reset_index()

    fixtures = fixtures.merge(
        stats_agg.rename(columns={
            "team_id": "home_team_id",
            "shots_total": "home_shots_total",
            "shots_on": "home_shots_on",
        }),
        on=["fixture_id", "home_team_id"],
        how="left"
    )
    fixtures = fixtures.merge(
        stats_agg.rename(columns={
            "team_id": "away_team_id",
            "shots_total": "away_shots_total",
            "shots_on": "away_shots_on",
        }),
        on=["fixture_id", "away_team_id"],
        how="left"
    )
    for col in ["home_shots_total", "home_shots_on", "away_shots_total", "away_shots_on"]:
        fixtures[col] = fixtures[col].fillna(0.0)

    history = build_team_elo_history(fixtures, rankings, base_goal_rate=5.0)

    if history.empty:
        return session.createDataFrame([], TEAM_ELO_SCHEMA)

    columns = [field.name for field in TEAM_ELO_SCHEMA.fields]
    return session.createDataFrame(history[columns], TEAM_ELO_SCHEMA)
