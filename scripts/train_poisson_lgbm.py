"""Train Poisson LightGBM models from the gold player event feature table.

One exposure-offset Poisson LightGBM model is trained per target event
(business_logic.md §2/§13.3) against fct_football__player_event_features.
Training rows use actual games_minutes as the exposure offset; inference uses
expected_minutes via ExposurePoissonLightGBMModel.predict_mean(exposure=...).

Example:
    python scripts/train_poisson_lgbm.py \
        --feature-table football_analytics.gold.fct_football__player_event_features \
        --registered-model-prefix football_analytics.gold.player_event
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from football_analytics.ml_training import (
    DEFAULT_TARGET_COLUMNS,
    PoissonLightGBMConfig,
    add_model_interaction_features,
    temporal_train_validation_split,
    train_poisson_lightgbm_with_mlflow,
)


REQUIRED_ELO_FEATURE_COLUMNS = [
    "team_elo_general_pre",
    "opponent_elo_general_pre",
    "team_elo_attack_pre",
    "team_elo_defense_pre",
    "opponent_elo_attack_pre",
    "opponent_elo_defense_pre",
    "expected_goals_for_pre",
    "expected_goals_against_pre",
    "player_offensive_modifier_pre",
    "player_defensive_modifier_pre",
    "player_offensive_elo_pre",
    "player_defensive_elo_pre",
    "player_offensive_rating_pre",
    "player_defensive_rating_pre",
    "missed_fixture_count_pre",
    "team_lineup_attack_strength",
    "team_lineup_defense_strength",
]


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _feature_table_version(spark, table_name: str) -> str | None:
    """Returns the current Delta version of the feature table, if readable."""

    try:
        history = spark.sql(f"DESCRIBE HISTORY {table_name} LIMIT 1").collect()
        if history:
            return str(history[0]["version"])
    except Exception:
        pass
    return None


def build_training_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    require_elo: bool = True,
) -> pd.DataFrame:
    """Validates the dbt-materialized feature table used for model training.

    fct_football__player_event_features also carries inference rows for
    future fixtures (null labels, is_completed_fixture = false); training must
    only see completed rows with real exposure minutes.
    """

    frame = feature_frame
    if "is_completed_fixture" in frame.columns:
        frame = frame[frame["is_completed_fixture"].fillna(False).astype(bool)]
    if "games_minutes" in frame.columns:
        frame = frame[pd.to_numeric(frame["games_minutes"], errors="coerce").fillna(0) > 0]

    if not require_elo:
        return frame.copy()

    missing = [
        column for column in REQUIRED_ELO_FEATURE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            "Feature table is missing dbt-materialized ELO columns: "
            + ", ".join(missing)
        )
    return add_model_interaction_features(frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-table",
        default="football_analytics.gold.fct_football__player_event_features",
        help="Fully-qualified Spark/Unity Catalog table containing training rows.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGET_COLUMNS),
        help="Comma-separated count targets to train independently.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Optional comma-separated feature override. Defaults to available built-ins.",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default="hierarchical-elo-poisson-lightgbm")
    parser.add_argument(
        "--registered-model-prefix",
        default=None,
        help=(
            "Unity Catalog prefix for registered models; each target registers "
            "as <prefix>__<target> (e.g. football_analytics.gold.player_event"
            "__shots_total)."
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--num-boost-round", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError("PySpark is required to load the Databricks feature table") from exc

    spark = SparkSession.builder.getOrCreate()

    if args.registered_model_prefix:
        # Registered models live in the Unity Catalog registry (see commit
        # "Configure MLflow to explicitly use Unity Catalog registry URI").
        import mlflow

        mlflow.set_registry_uri("databricks-uc")

    feature_frame = spark.table(args.feature_table).toPandas()
    training_frame = build_training_feature_frame(feature_frame)
    train_df, validation_df = temporal_train_validation_split(
        training_frame,
        validation_fraction=args.validation_fraction,
    )

    config = PoissonLightGBMConfig(
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        num_boost_round=args.num_boost_round,
    )
    feature_table_version = _feature_table_version(spark, args.feature_table)
    result = train_poisson_lightgbm_with_mlflow(
        train_df,
        validation_df,
        target_columns=_parse_csv(args.targets) or DEFAULT_TARGET_COLUMNS,
        feature_columns=_parse_csv(args.features),
        config=config,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        registered_model_name_prefix=args.registered_model_prefix,
        run_tags={
            "feature_table_name": args.feature_table,
            "feature_table_version": feature_table_version or "unknown",
        },
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(json.dumps({"feature_columns": result["feature_columns"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
